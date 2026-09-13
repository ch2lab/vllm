import sys

p = 'csrc/libtorch_stable/attention/flash_attn_sm70_wmma.cu'
s = open(p).read()

old_helper = """__device__ __forceinline__ half fp8e4m3_to_half_scaled(uint8_t x,
                                                       float scale) {
  __half_raw hr = __nv_cvt_fp8_to_halfraw(x, __NV_E4M3);
  float val = __half2float(*reinterpret_cast<half*>(&hr)) * scale;
  return __float2half(val);
}"""
new_helper = old_helper + """

__device__ __forceinline__ half fp8e5m2_to_half_scaled(uint8_t x,
                                                       float scale) {
  __half_raw hr = __nv_cvt_fp8_to_halfraw(x, __NV_E5M2);
  float val = __half2float(*reinterpret_cast<half*>(&hr)) * scale;
  return __float2half(val);
}"""
assert old_helper in s
s = s.replace(old_helper, new_helper)

s = s.replace(
    "// KV_MODE: 0 = FP16, 1 = FP8 (e4m3, dequant inline), 2 = INT4, 3 = NVFP4",
    "// KV_MODE: 0 = FP16, 1 = FP8 e4m3, 2 = INT4, 3 = NVFP4, 4 = FP8 e5m2")

for scale_name, off in (("k_scale", "+ d8"), ("v_scale", "+ HD + d8")):
    old_call = f"sKV[n][d8 + i] = fp8e4m3_to_half_scaled(bytes[i], {scale_name});"
    new_call = (
        "sKV[n][d8 + i] = (KV_MODE == 4)\n"
        f"                ? fp8e5m2_to_half_scaled(bytes[i], {scale_name})\n"
        f"                : fp8e4m3_to_half_scaled(bytes[i], {scale_name});")
    n = s.count(old_call)
    assert n == 1, (scale_name, n)
    s = s.replace(old_call, new_call)

# The two blocks above are inside `if constexpr (KV_MODE == 1)`; widen them.
# They are distinguishable by the base-offset line (+d8 vs +HD+d8 already
# handled); widen the branch header that precedes each. We simply replace
# all occurrences of the exact header used in the paged kernel.
s = s.replace("if constexpr (KV_MODE == 1) {",
              "if constexpr (KV_MODE == 1 || KV_MODE == 4) {")

# Launcher dispatch: add mode 4 next to mode 1.
for hd in (128, 256):
    old = (f"if (mode == 2) {{ LAUNCH_PARTED(2, {hd}); }}\n"
           f"    else if (mode == 3) {{ LAUNCH_PARTED(3, {hd}); }}\n"
           f"    else if (mode == 1) {{ LAUNCH_PARTED(1, {hd}); }}")
    new = (f"if (mode == 2) {{ LAUNCH_PARTED(2, {hd}); }}\n"
           f"    else if (mode == 3) {{ LAUNCH_PARTED(3, {hd}); }}\n"
           f"    else if (mode == 4) {{ LAUNCH_PARTED(4, {hd}); }}\n"
           f"    else if (mode == 1) {{ LAUNCH_PARTED(1, {hd}); }}")
    assert old in s, hd
    s = s.replace(old, new)

# Batched prefill macro: the LAUNCH_BATCHED_KERNEL chain lives inside a
# #define; extend it for mode 4 by matching the two-line pattern.
bs = "    else if (mode == 1) { LAUNCH_BATCHED_KERNEL(1, HD_VAL) }"
assert bs in s
s = s.replace(
    bs,
    "    else if (mode == 4) { LAUNCH_BATCHED_KERNEL(4, HD_VAL) } \\\n" + bs)

open(p, 'w').write(s)
print('patched kernel OK')
