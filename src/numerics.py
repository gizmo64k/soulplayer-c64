#!/usr/bin/env python3
"""
BOB v3 NUMERICS — proper fixed-point with per-tensor scales.

Activations: Q8.8 signed int16 (int_val = round(float_val * 256))
Weights:     int8 with per-tensor integer shift s_w such that
             int8_w = clamp(round(float_w * 2^s_w), -127, 127)
             s_w chosen so max-abs weight maps near 127.
Biases:      stored as int32 pre-scaled to match the matmul accumulator,
             so `acc += bias_q32` with no per-bias shift at runtime.

Matmul: acc_i32 = Σ (x_q88 * W_int8);  out_q88 = acc_i32 >> s_w
RMSNorm: direct fixed-point implementation matching the 6502 version.
Softmax: float internally here; the 6502 will use an exp LUT calibrated
         to produce the same output.

This file is the ground truth. Shadow, 6502 port, and tests must match it.
"""
import numpy as np
import math

VS = 128
ED = 32
NH = 4
HD = 8
FF = 64
NL = 2
SL = 20

PAD, SEP, UNK, END = 0, 1, 2, 3

ACT_SHIFT = 8  # Q8.8


def pick_shift(abs_max: float) -> int:
    if abs_max <= 1e-12:
        return 0
    return int(math.floor(math.log2(127.0 / abs_max)))


def quantize_i8(float_arr: np.ndarray, shift=None):
    abs_max = float(np.abs(float_arr).max())
    if shift is None:
        shift = pick_shift(abs_max) if abs_max > 0 else 0
    q = np.round(float_arr * (2.0 ** shift))
    q = np.clip(q, -127, 127).astype(np.int8)
    return q, shift


def pack_tensor(float_arr, shift=None):
    q, s = quantize_i8(float_arr, shift=shift)
    return {'q': q, 's': s}


def pack_bias(float_bias, matmul_shift):
    """Pre-scale a bias to match the matmul accumulator's internal scale.
    Matmul accumulator for row r is Σ x_q88[c] * W_int8[r,c], which is
    a float*2^(8+s_w) value. So bias_q = round(bias_float * 2^(8+s_w)).

    Empirically these values fit comfortably in int16 for our model, so we
    store them as int16 to save memory on the 6502 (saves 1KB vs int32).
    We saturate defensively.
    """
    scale = 1 << (ACT_SHIFT + matmul_shift)
    q = np.round(float_bias * scale)
    q = np.clip(q, -32768, 32767).astype(np.int16)
    return {'q16': q, 's': matmul_shift}


class Weights:
    def __init__(self):
        self.te = None
        self.pe = None
        self.layers = [None] * NL
        self.norm_w = None
        self.out_w = None


def sat16(v):
    if v > 32767:  return 32767
    if v < -32768: return -32768
    return int(v)

def sat16_arr(a):
    return np.clip(a, -32768, 32767).astype(np.int16)


def matvec(W_entry, x_q88, post_shift=1):
    """
    Fixed-point matrix-vector multiply.
    post_shift: extra right shift applied after s_w. Set to 1 for any
    matmul trained with QATLinear(post_shift=True) — q, k, v, proj, fc1, fc2.
    Set to 0 for the output projection (and anything else with post_shift=False).
    """
    W = W_entry['q'].astype(np.int32)
    s_w = W_entry['s']
    x = x_q88.astype(np.int32)
    acc = (W * x[np.newaxis, :]).sum(axis=1)
    total = s_w + post_shift
    if total >= 0: out = acc >> total
    else:          out = acc << (-total)
    return sat16_arr(out)


def matvec_bias(W_entry, bias_entry, x_q88, post_shift=1):
    W = W_entry['q'].astype(np.int32)
    s_w = W_entry['s']
    x = x_q88.astype(np.int32)
    acc = (W * x[np.newaxis, :]).sum(axis=1) + bias_entry['q16'].astype(np.int32)
    total = s_w + post_shift
    if total >= 0: out = acc >> total
    else:          out = acc << (-total)
    return sat16_arr(out)


def isqrt_u32(v: int) -> int:
    """Integer square root of an unsigned 32-bit value, returns floor(sqrt(v)).
    Matches the bit-by-bit algorithm the 6502 will use."""
    if v <= 0:
        return 0
    # Find the highest bit of the result. Start with the top "odd bit" under v.
    result = 0
    bit = 1 << 30   # second-highest bit (bit 30), largest power of 4 <= 2^31
    while bit > v:
        bit >>= 2
    while bit != 0:
        if v >= result + bit:
            v -= result + bit
            result = (result >> 1) + bit
        else:
            result >>= 1
        bit >>= 2
    return result


def udiv_u32_u16(num: int, den: int) -> int:
    """Unsigned 32/16 → 16 divide with truncation (floor). Matches what the
    6502 restoring-division routine computes."""
    if den == 0:
        return 0xFFFF
    q = num // den
    if q > 0xFFFF:
        return 0xFFFF
    return q


def rms_norm(x_q88, gain_entry):
    """RMSNorm on Q8.8 input, int8 gain with shift. Output Q8.8.
    Uses integer sqrt and integer divide so the 6502 port can match bit-exact.

    INV is computed as 2^19 / rms (not 2^20) so it stays ≤ 32767 and can be
    used as a signed int16 in smul16. The matching shift is >> 15 instead of >> 16.
    """
    n = len(x_q88)
    xs = x_q88.astype(np.int32) >> 4
    sum_sq = int((xs * xs).sum())
    mean_sq = max(1, sum_sq // n)
    rms_shifted = isqrt_u32(mean_sq)
    if rms_shifted < 1:
        rms_shifted = 1
    inv = udiv_u32_u16(1 << 19, rms_shifted)
    if inv > 32767:
        inv = 32767
    g = gain_entry['q'].astype(np.int32)
    s_g = gain_entry['s']
    out = np.empty(n, dtype=np.int16)
    for i in range(n):
        y_raw = (int(x_q88[i]) * inv) >> 15
        y = (y_raw * int(g[i])) >> s_g
        out[i] = sat16(y)
    return out


def relu_q88(x):
    return np.maximum(x, 0).astype(np.int16)


# Exp LUT for softmax. EXP_LUT[i] = round(255 * exp(-i / 16)) for i in 0..127.
# Calibration: one "LUT unit" = 1/16 of a float-exponent unit. Score differences
# up to ~80 LUT units produce meaningful weights (down to ~1 in 255); beyond
# that the weight is 0.
EXP_LUT = np.zeros(128, dtype=np.uint8)
for _i in range(128):
    _v = 255.0 * math.exp(-_i / 16.0)
    EXP_LUT[_i] = max(1, min(255, int(round(_v)))) if _i > 0 else 255
EXP_LUT[127] = 0  # guard: anything this far below max is zero


def softmax_weighted_sum(scores_i32, values_q88):
    """
    Integer softmax via exp LUT + weighted sum + divide.

    Matches exactly what the 6502 will compute.

    Steps:
      1. Normalize scores: divide by 2^17 (rough sqrt(HD) scaling), giving
         signed int16 values sf[i].
      2. Find max_sf.
      3. For each i: delta = max_sf - sf[i] (always ≥ 0). Clamp to [0, 127].
         weight[i] = EXP_LUT[delta], a u8.
      4. w_sum = Σ weight[i], u16.
      5. For each output dim j: acc[j] = Σ weight[i] * values[i, j]  (int32)
      6. out[j] = acc[j] // w_sum, saturated to int16.
    """
    n = len(scores_i32)
    # Step 1: sf[i] = scores[i] >> 14
    # The shift determines how much of the score dynamic range maps into the
    # LUT's 128 entries. >>17 was too aggressive — it crushed all score
    # differences to ~5 levels, making attention essentially uniform.
    # >>14 gives ~35-70 LUT units of spread, matching float softmax closely.
    sf = scores_i32.astype(np.int64) >> 14
    # Step 2
    max_sf = int(sf.max())
    # Step 3: build weights via LUT
    weights = np.empty(n, dtype=np.uint8)
    for i in range(n):
        delta = max_sf - int(sf[i])
        if delta < 0:   delta = 0       # shouldn't happen; guard
        if delta > 127: delta = 127
        weights[i] = EXP_LUT[delta]
    # Step 4
    w_sum = int(weights.sum())
    if w_sum == 0:
        w_sum = 1
    # Step 5 & 6
    HD_ = values_q88.shape[1]
    out = np.empty(HD_, dtype=np.int16)
    for j in range(HD_):
        acc = 0
        for i in range(n):
            acc += int(weights[i]) * int(values_q88[i, j])
        q = acc // w_sum
        out[j] = sat16(q)
    return out


def forward(W: Weights, token_ids):
    T = len(token_ids)
    assert 1 <= T <= SL

    # Embedding: convert stored int8 back to Q8.8
    def deshift(v, s):
        diff = ACT_SHIFT - s
        if diff >= 0: return v.astype(np.int32) << diff
        else:         return v.astype(np.int32) >> (-diff)

    te_q, s_te = W.te['q'], W.te['s']
    pe_q, s_pe = W.pe['q'], W.pe['s']

    h = np.zeros((T, ED), dtype=np.int16)
    for t, tok in enumerate(token_ids):
        v = deshift(te_q[tok], s_te) + deshift(pe_q[t], s_pe)
        h[t] = sat16_arr(v)

    for L in range(NL):
        lay = W.layers[L]
        # Per-position Q/K/V with pre-norm
        q_all = np.zeros((T, ED), dtype=np.int16)
        k_all = np.zeros((T, ED), dtype=np.int16)
        v_all = np.zeros((T, ED), dtype=np.int16)
        for t in range(T):
            xn = rms_norm(h[t], lay['n1'])
            q_all[t] = matvec(lay['q'], xn)
            k_all[t] = matvec(lay['k'], xn)
            v_all[t] = matvec(lay['v'], xn)

        # Causal attention at every position
        att_new = np.zeros((T, ED), dtype=np.int16)
        for t_q in range(T):
            for head in range(NH):
                sl = slice(head * HD, (head + 1) * HD)
                q_vec = q_all[t_q, sl]
                k_head = k_all[:t_q + 1, sl]
                v_head = v_all[:t_q + 1, sl]
                n_keys = t_q + 1
                scores = np.zeros(n_keys, dtype=np.int32)
                for t_k in range(n_keys):
                    scores[t_k] = int((q_vec.astype(np.int32) *
                                       k_head[t_k].astype(np.int32)).sum())
                att_new[t_q, sl] = softmax_weighted_sum(scores, v_head)

        # Output projection per position with residual
        for t in range(T):
            att_proj = matvec(lay['proj'], att_new[t])
            h[t] = sat16_arr(h[t].astype(np.int32) + att_proj.astype(np.int32))

        # FFN per position with residual
        for t in range(T):
            yn = rms_norm(h[t], lay['n2'])
            z  = matvec_bias(lay['fc1_w'], lay['fc1_b'], yn)
            z  = relu_q88(z)
            w2 = matvec_bias(lay['fc2_w'], lay['fc2_b'], z)
            h[t] = sat16_arr(h[t].astype(np.int32) + w2.astype(np.int32))

    # Final norm + logits
    y = rms_norm(h[T - 1], W.norm_w)
    logits = matvec(W.out_w, y, post_shift=0)

    best = 4
    best_v = int(logits[4])
    for i in range(5, VS):
        if int(logits[i]) > best_v:
            best_v = int(logits[i])
            best = i
    return best, logits
