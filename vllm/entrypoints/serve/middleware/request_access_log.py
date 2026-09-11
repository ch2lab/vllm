# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
from collections.abc import Awaitable
from http import HTTPStatus

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from vllm.logger import init_logger

logger = init_logger(__name__)


class RequestAccessLogMiddleware:
    """Emit one access-log line per HTTP request, replacing uvicorn's built-in
    access log.

    The line is ``client - "METHOD PATH HTTP/ver" STATUS [reason]``; for
    requests whose serving handler stashed per-request metrics (chat /
    completions / responses), it is extended with
    ``| request_id: prompt_tokens=... ...``. The proxy-aware resolved client
    is used when available, else the direct TCP peer. When the caller's
    User-Agent is known (nginx forwards it in ``X-Forwarded-User-Agent``),
    it is shown after the client as ``client [ua]``.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    def __call__(self, scope: Scope, receive: Receive, send: Send) -> Awaitable[None]:
        if scope["type"] != "http":
            return self.app(scope, receive, send)

        status = {"code": 0}

        async def send_wrapped(message: Message) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
            elif message["type"] == "http.response.body" and not message.get(
                "more_body", False
            ):
                self._log(scope, status["code"])
            await send(message)

        return self.app(scope, receive, send_wrapped)

    def _log(self, scope: Scope, code: int) -> None:
        if not logger.isEnabledFor(logging.INFO):
            return
        try:
            logger.info("%s", self._format_line(scope, code))
        except Exception:
            # Never let an access-log failure break request handling.
            logger.debug("Failed to write request access log.", exc_info=True)

    def _format_line(self, scope: Scope, code: int) -> str:
        state = scope.get("state") or {}
        metadata = state.get("request_metadata")

        if metadata is not None and metadata.client:
            client = metadata.client
        else:
            peer = scope.get("client")
            client = f"{peer[0]}:{peer[1]}" if peer else "-"

        path = scope.get("path", "")
        query = scope.get("query_string", b"")
        if query:
            path = f"{path}?{query.decode('latin-1')}"

        protocol = f"HTTP/{scope.get('http_version', '1.1')}"
        reason = HTTPStatus(code).phrase if 0 < code < 600 else ""
        ua = self._user_agent(scope)
        caller = f"{client} [{ua}]" if ua else client
        line = f'{caller} - "{scope.get("method", "")} {path} {protocol}" {code}'
        if reason:
            line = f"{line} {reason}"

        if metadata is not None and metadata.summary:
            line = f"{line} | {metadata.request_id}: {metadata.summary}"
        return line

    @staticmethod
    def _user_agent(scope: Scope) -> str | None:
        # Prefer the edge client's UA that nginx forwards in
        # ``X-Forwarded-User-Agent`` (litellm relays x-* headers downstream);
        # fall back to the request's own ``User-Agent``.
        for name in (b"x-forwarded-user-agent", b"user-agent"):
            for key, value in scope.get("headers", []):
                if key.lower() == name:
                    return value.decode("latin-1")
        return None
