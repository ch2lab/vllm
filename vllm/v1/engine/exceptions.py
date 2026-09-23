# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from asyncio import CancelledError
from typing import TYPE_CHECKING

from vllm.exceptions import VLLMServerError

if TYPE_CHECKING:
    from vllm.v1.metrics.stats import AbortedRequestStats


class EngineGenerateError(VLLMServerError):
    """Raised when a AsyncLLM.generate() fails. Recoverable."""

    pass


class EngineDeadError(VLLMServerError):
    """Raised when the EngineCore dies. Unrecoverable."""

    def __init__(self, *args, suppress_context: bool = False, **kwargs):
        ENGINE_DEAD_MESSAGE = "EngineCore encountered an issue. See stack trace (above) for the root cause."  # noqa: E501

        super().__init__(ENGINE_DEAD_MESSAGE, *args, **kwargs)
        # Make stack trace clearer when using with LLMEngine by
        # silencing irrelevant ZMQError.
        self.__suppress_context__ = suppress_context


class ClientDisconnectedError(CancelledError):
    """Cancellation raised when a client disconnects mid-request.

    Carries the per-child live stats snapshotted before abort cleanup, so
    the serving layer can still report metrics for requests (e.g.
    non-streaming) that never received an output to build one from.
    """

    def __init__(self, aborted_stats: "list[AbortedRequestStats]"):
        super().__init__("client disconnected")
        self.aborted_stats = aborted_stats
