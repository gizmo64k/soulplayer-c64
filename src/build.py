#!/usr/bin/env python3
"""
SOUL C64 BUILDER v6.4
=========================
Real transformer. Real attention. Real softmax. Real RMSNorm.
25KB soul. 1 MHz. Love.

Assembles all proven-correct 6502 routines into a single PRG file,
embeds the trained int8 weights, wraps in a D64 disk image.

Usage:
    python3 build_c64_v6.py

Requires in the same directory:
    bob_soul_v3.bin        (from reexport_v3.py)
    bob_tokenizer_v2.json  (from bob_transformer_v2.py)

Outputs:
    soulplayer.prg          PRG file (load at $0801)
    soulplayer.d64          D64 disk image for VICE / real hardware
"""
import struct, json, os, sys
import numpy as np

from assembler import CodeBuilder
from asm_matvec import build_matvec, WP, SP, DP, ROWS, COLS, SHIFT, BP, BFLAG
from asm_rms_norm import (
    build_rms_norm, XP as RMS_XP, GP as RMS_GP, DP as RMS_DP, SG as RMS_SG,
    SUMSQ, IDX, TMP, SRC16, SIGN, PROD, RMS, INV, T32, SCR_A, SCR_B,
)
from asm_attn_head import (
    build_attn_head, build_sdiv_i32_u16, build_smul16,
    QP, KB, VB, OP, NKEYS, HEAD,
    SCORES_P, WTS_P, ELUT_P, MAXSF, WSUM, TIDX, J, KROW,
)
from asm_simple import (
    build_embed_one_v2, build_argmax,
    TP, PP, SH1, SH2,
)
from numerics import VS, ED, NH, HD, FF, NL, SL, EXP_LUT, ACT_SHIFT
from soul_io import tensor_spec, read_soul_v3

# ─── Constants ──────────────────────────────────────────────────────
CHROUT = 0xFFD2
CHRIN  = 0xFFCF

SEP_TOK = 1
END_TOK = 3
PAD_TOK = 0

MAXSEQ = SL    # 20 (but we may use 16 at runtime for speed)

# ─── Memory layout ─────────────────────────────────────────────────
# Everything must live below $A000 (BASIC ROM starts there in default bank).
# Layout:
#   $0801-$1FEA : code + data tables
#   $2000-$8260 : weights (25.3KB, loaded from PRG)
#   $8280-$9D80 : activation buffers (~6.8KB)
#   $C000-$C1FF : token buffers + LOGITS_BUF (always-RAM region)
STRIDE = ED * 2    # 64 bytes per position

WEIGHTS_ADDR = 0x2100   # right after code (with extra room for polish)

# Buffers placed after weights. Weights end at $85A0 ($2100 + 25760).
# BUF_BASE = $8600 puts HIDDEN safely after weights end with no overlap.
# The 5 big per-position buffers fit in $8600-$9F00 (5 × $500 = $1900), all below $A000.
# SCRATCH lives separately at $C200 in always-RAM (after TOKS/INPUT/LOGITS_BUF)
# because extending the main buffer region past $9F00 crosses into BASIC ROM.
BUF_BASE = 0x8600
HIDDEN   = BUF_BASE + 0x0000   # 1280 bytes (SL*STRIDE)
Q_ALL    = BUF_BASE + 0x0500
K_ALL    = BUF_BASE + 0x0A00
V_ALL    = BUF_BASE + 0x0F00
ATT_OUT  = BUF_BASE + 0x1400
# End of main buffers: $9F00 (safely below $A000)
SCRATCH  = 0xC200              # 336 bytes in always-RAM region

# Within SCRATCH
SCORES_BUF = SCRATCH + 0x000   # SL * 2 = 40 bytes (int16 sf values)
WEIGHTS_BUF = SCRATCH + 0x030  # SL bytes
XN_BUF   = SCRATCH + 0x050     # ED * 2 = 64 bytes (normed hidden)
Z_BUF    = SCRATCH + 0x090     # FF * 2 = 128 bytes (fc1 output)
W2_BUF   = SCRATCH + 0x110     # ED * 2 = 64 bytes (fc2 output / proj output)

# Token buffer (input/output token IDs)
# Token buffer — MUST NOT be in screen RAM ($0400-$07E7) or under BASIC ROM.
# $C000-$CFFF is 4KB of always-RAM with no ROM or I/O shadowing.
TOKS     = 0xC000              # 32 bytes
SLEN     = 0xC030              # current sequence length
GPOS     = 0xC031              # generation position counter

# Input text buffer
INPUT    = 0xC040              # 64 bytes


# ─── Tokenizer tables ──────────────────────────────────────────────
def build_tokenizer_tables(tok_path):
    tok = json.load(open(tok_path))
    vocab = tok['vocab']
    merges = tok['merges']
    id_to_str = {v: k for k, v in vocab.items()}
    strings = bytearray()
    offsets = []
    for i in range(128):
        offsets.append(len(strings))
        s = id_to_str.get(i, '')
        if s.startswith('<'): s = ''
        for ch in s:
            strings.append(ord(ch) & 0x7F)
        strings.append(0)
    offset_bytes = bytearray()
    for o in offsets:
        offset_bytes.extend(struct.pack('<H', o))
    merge_bytes = bytearray()
    for a, b in merges:
        ta = vocab.get(a)
        tb = vocab.get(b)
        tm = vocab.get(a + b)
        if ta is not None and tb is not None and tm is not None:
            merge_bytes.extend([ta & 0xFF, tb & 0xFF, tm & 0xFF])
    merge_bytes.append(0xFF)
    return bytes(offset_bytes), bytes(strings), bytes(merge_bytes)


# ─── Parse soul file for raw weight data ───────────────────────────
def parse_soul_for_c64(path):
    """Read the v3 soul file and return:
       - raw_blob: concatenated raw bytes (int8 weights + int16 biases)
       - tensor_info: list of (name, kind, offset_in_blob, size, shift)
    """
    data = open(path, 'rb').read()
    off = 0
    blob = bytearray()
    info = []
    for name, kind, shape in tensor_spec():
        k, rows, cols, s = struct.unpack_from('<BHHb', data, off)
        off += 6
        n = rows * cols
        blob_off = len(blob)
        if k == 0:  # int8
            raw = data[off:off + n]
            off += n
            blob.extend(raw)
            info.append((name, 'w', blob_off, n, s))
        else:       # int16 bias
            raw = data[off:off + n * 2]
            off += n * 2
            blob.extend(raw)
            info.append((name, 'b', blob_off, n * 2, s))
    return bytes(blob), info


# ─── Build the program ─────────────────────────────────────────────
def build_program(soul_blob, tensor_info, tok_offsets, tok_strings, tok_merges):
    cb = CodeBuilder(0x0801)

    # ═══ BASIC stub ═══
    cb.emit(0x0C, 0x08, 0x0A, 0x00, 0x9E, 0x20)
    cb.emit(0x32, 0x30, 0x36, 0x32)  # "2062"
    cb.emit(0x00, 0x00, 0x00)

    # ═══ Entry point ($080E) ═══
    cb.label('entry')
    cb.emit(0xA9, 0x00, 0x8D, 0x21, 0xD0)  # black bg
    cb.emit(0xA9, 0x05, 0x8D, 0x20, 0xD0)  # green border
    cb.emit(0xA9, 0x93); cb.emit(0x20, CHROUT & 0xFF, (CHROUT >> 8) & 0xFF)
    cb.emit(0xA9, 0x0E); cb.emit(0x20, CHROUT & 0xFF, (CHROUT >> 8) & 0xFF)
    cb.emit(0xA9, 0x1E); cb.emit(0x20, CHROUT & 0xFF, (CHROUT >> 8) & 0xFF)

    # Print banner
    cb.emit(0xA9); cb.patches.append((cb.foff(), 'banner', 'lobyte')); cb.emit(0)
    cb.emit(0xA0); cb.patches.append((cb.foff(), 'banner', 'hibyte')); cb.emit(0)
    cb.emit_jsr('print_str')
    cb.emit(0xA9); cb.patches.append((cb.foff(), 'ready_msg', 'lobyte')); cb.emit(0)
    cb.emit(0xA0); cb.patches.append((cb.foff(), 'ready_msg', 'hibyte')); cb.emit(0)
    cb.emit_jsr('print_str')

    # ═══ Main loop ═══
    cb.label('main_loop')
    cb.emit(0xA9, 0x0D); cb.emit(0x20, CHROUT & 0xFF, (CHROUT >> 8) & 0xFF)
    cb.emit(0xA9); cb.patches.append((cb.foff(), 'prompt_str', 'lobyte')); cb.emit(0)
    cb.emit(0xA0); cb.patches.append((cb.foff(), 'prompt_str', 'hibyte')); cb.emit(0)
    cb.emit_jsr('print_str')
    cb.emit_jsr('read_line')
    # Check quit
    cb.emit(0xAD, INPUT & 0xFF, (INPUT >> 8) & 0xFF)
    cb.emit(0xC9, 0x51)  # 'Q'
    cb.emit_branch_far(0xF0, 'quit')
    cb.emit(0xC9, 0x71)  # 'q'
    cb.emit_branch_far(0xF0, 'quit')
    cb.emit_jsr('encode_input')
    cb.emit_jsr('run_inference')
    cb.emit_jmp('main_loop')

    cb.label('quit')
    cb.emit(0xA9, 0x0D); cb.emit(0x20, CHROUT & 0xFF, (CHROUT >> 8) & 0xFF)
    cb.emit(0xA9); cb.patches.append((cb.foff(), 'quit_msg', 'lobyte')); cb.emit(0)
    cb.emit(0xA0); cb.patches.append((cb.foff(), 'quit_msg', 'hibyte')); cb.emit(0)
    cb.emit_jsr('print_str')
    cb.emit(0x60)  # RTS

    # ═══ I/O routines (from v3, proven) ═══

    # print_str: A=lo, Y=hi
    cb.label('print_str')
    cb.emit(0x85, 0x50, 0x84, 0x51, 0xA0, 0x00)
    cb.label('_ps_lp')
    cb.emit(0xB1, 0x50)
    cb.emit_branch(0xF0, '_ps_done')
    cb.emit(0x20, CHROUT & 0xFF, (CHROUT >> 8) & 0xFF)
    cb.emit(0xC8)
    cb.emit_jmp('_ps_lp')
    cb.label('_ps_done')
    cb.emit(0x60)

    # read_line: reads into INPUT buffer
    cb.label('read_line')
    cb.emit(0xA2, 0x00)
    cb.label('_rl_lp')
    cb.emit(0x20, CHRIN & 0xFF, (CHRIN >> 8) & 0xFF)
    cb.emit(0xC9, 0x0D)
    cb.emit_branch(0xF0, '_rl_done')
    cb.emit(0xE0, 0x3E)
    cb.emit_branch(0xB0, '_rl_skip')
    cb.emit(0x9D, INPUT & 0xFF, (INPUT >> 8) & 0xFF)
    cb.emit(0xE8)
    cb.label('_rl_skip')
    cb.emit_jmp('_rl_lp')
    cb.label('_rl_done')
    cb.emit(0xA9, 0x00)
    cb.emit(0x9D, INPUT & 0xFF, (INPUT >> 8) & 0xFF)
    cb.emit(0x60)

    # encode_input: convert INPUT text to token IDs in TOKS
    cb.label('encode_input')
    cb.emit(0xA9, SEP_TOK)
    cb.emit(0x8D, TOKS & 0xFF, (TOKS >> 8) & 0xFF)
    cb.emit(0xA2, 0x00, 0xA0, 0x01)
    cb.label('_enc_lp')
    cb.emit(0xBD, INPUT & 0xFF, (INPUT >> 8) & 0xFF)
    cb.emit_branch(0xF0, '_enc_zero')
    cb.emit(0xC9, 0x20)
    cb.emit_branch(0xD0, '_enc_notsp')
    cb.emit(0xA9, 0x04)
    cb.emit_jmp('_enc_store')
    cb.label('_enc_notsp')
    cb.emit(0xC9, 0x41)
    cb.emit_branch(0x90, '_enc_punct')
    cb.emit(0xC9, 0x5B)
    cb.emit_branch(0xB0, '_enc_punct')
    cb.emit(0x38, 0xE9, 0x3C)
    cb.emit_jmp('_enc_store')
    cb.label('_enc_punct')
    cb.emit(0xC9, 0x2E)                           # '.'
    cb.emit_branch(0xD0, '_enc_p2')
    cb.emit(0xA9, 31); cb.emit_jmp('_enc_store')
    cb.label('_enc_p2')
    cb.emit(0xC9, 0x3F)                           # '?'
    cb.emit_branch(0xD0, '_enc_p3')
    cb.emit(0xA9, 34); cb.emit_jmp('_enc_store')
    cb.label('_enc_p3')
    cb.emit(0xC9, 0x21)                           # '!'
    cb.emit_branch(0xD0, '_enc_p4')
    cb.emit(0xA9, 33); cb.emit_jmp('_enc_store')
    cb.label('_enc_p4')
    cb.emit(0xC9, 0x27)                           # apostrophe
    cb.emit_branch(0xD0, '_enc_p5')
    cb.emit(0xA9, 32); cb.emit_jmp('_enc_store')
    cb.label('_enc_p5')
    cb.emit(0xC9, 0x2C)                           # ','
    cb.emit_branch(0xD0, '_enc_p6')
    cb.emit(0xA9, 35); cb.emit_jmp('_enc_store')
    cb.label('_enc_p6')
    cb.emit(0xC9, 0x3B)                           # ';'
    cb.emit_branch(0xD0, '_enc_p7')
    cb.emit(0xA9, 36); cb.emit_jmp('_enc_store')
    cb.label('_enc_p7')
    cb.emit(0xC9, 0x3A)                           # ':'
    cb.emit_branch(0xD0, '_enc_p8')
    cb.emit(0xA9, 37); cb.emit_jmp('_enc_store')
    cb.label('_enc_p8')
    cb.emit(0xC9, 0x2D)                           # '-'
    cb.emit_branch(0xD0, '_enc_skip')
    cb.emit(0xA9, 38); cb.emit_jmp('_enc_store')
    cb.label('_enc_skip')
    cb.emit_jmp('_enc_lp')                        # unknown char → skip
    cb.label('_enc_store')
    cb.emit(0x99, TOKS & 0xFF, (TOKS >> 8) & 0xFF)
    cb.emit(0xC8, 0xE8)
    cb.emit(0xC0, MAXSEQ - 2)
    cb.emit_branch_far(0xB0, '_enc_done')
    cb.emit_jmp('_enc_lp')
    cb.label('_enc_zero')
    cb.label('_enc_done')
    cb.emit(0xA9, SEP_TOK)
    cb.emit(0x99, TOKS & 0xFF, (TOKS >> 8) & 0xFF)
    cb.emit(0xC8)
    cb.emit(0x8C, SLEN & 0xFF, (SLEN >> 8) & 0xFF)
    cb.emit_jsr('apply_bpe')
    cb.emit(0x60)

    # apply_bpe
    cb.label('apply_bpe')
    cb.emit(0xA9); cb.patches.append((cb.foff(), 'merge_table', 'lobyte')); cb.emit(0)
    cb.emit(0x85, 0x50)
    cb.emit(0xA9); cb.patches.append((cb.foff(), 'merge_table', 'hibyte')); cb.emit(0)
    cb.emit(0x85, 0x51)
    cb.label('_bpe_next')
    cb.emit(0xA0, 0x00)
    cb.emit(0xB1, 0x50)
    cb.emit(0xC9, 0xFF); cb.emit_branch(0xD0, '_bpe_cont')
    cb.emit_jmp('_bpe_ret')
    cb.label('_bpe_cont')
    cb.emit(0x85, 0x02)
    cb.emit(0xC8, 0xB1, 0x50, 0x85, 0x03)
    cb.emit(0xC8, 0xB1, 0x50, 0x85, 0x04)
    cb.emit(0xA2, 0x00)
    cb.label('_bpe_scan')
    cb.emit(0xEC, SLEN & 0xFF, (SLEN >> 8) & 0xFF)
    cb.emit_branch_far(0xB0, '_bpe_adv')
    cb.emit(0xBD, TOKS & 0xFF, (TOKS >> 8) & 0xFF)
    cb.emit(0xC5, 0x02)
    cb.emit_branch_far(0xD0, '_bpe_np')
    cb.emit(0x8A, 0x48, 0xE8)
    cb.emit(0xBD, TOKS & 0xFF, (TOKS >> 8) & 0xFF)
    cb.emit(0x85, 0x05)
    cb.emit(0x68, 0xAA)
    cb.emit(0xA5, 0x05, 0xC5, 0x03)
    cb.emit_branch_far(0xD0, '_bpe_np')
    cb.emit(0xA5, 0x04)
    cb.emit(0x9D, TOKS & 0xFF, (TOKS >> 8) & 0xFF)
    cb.emit(0x86, 0x05)
    cb.emit(0xE8, 0xE8)
    cb.label('_bpe_shift')
    cb.emit(0xEC, SLEN & 0xFF, (SLEN >> 8) & 0xFF)
    cb.emit_branch_far(0xB0, '_bpe_shifted')
    cb.emit(0xBD, TOKS & 0xFF, (TOKS >> 8) & 0xFF)
    cb.emit(0xCA)
    cb.emit(0x9D, TOKS & 0xFF, (TOKS >> 8) & 0xFF)
    cb.emit(0xE8, 0xE8)
    cb.emit_jmp('_bpe_shift')
    cb.label('_bpe_shifted')
    cb.emit(0xCE, SLEN & 0xFF, (SLEN >> 8) & 0xFF)
    cb.emit(0xA6, 0x05)
    cb.emit_jmp('_bpe_scan')
    cb.label('_bpe_np')
    cb.emit(0xE8)
    cb.emit_jmp('_bpe_scan')
    cb.label('_bpe_adv')
    cb.emit(0x18, 0xA5, 0x50, 0x69, 0x03, 0x85, 0x50)
    cb.emit(0x90, 0x02, 0xE6, 0x51)
    cb.emit_jmp('_bpe_next')
    cb.label('_bpe_ret')
    cb.emit(0x60)

    # print_token: decode token A and print
    cb.label('print_token')
    cb.emit(0x0A, 0xA8)
    cb.emit(0xB9); cb.patches.append((cb.foff(), 'decode_offsets', 'abs16')); cb.emit(0, 0)
    cb.emit(0x85, 0x50)
    cb.emit(0xC8)
    cb.emit(0xB9); cb.patches.append((cb.foff(), 'decode_offsets', 'abs16')); cb.emit(0, 0)
    cb.emit(0x85, 0x51)
    cb.emit(0x18, 0xA5, 0x50)
    cb.emit(0x69); cb.patches.append((cb.foff(), 'decode_strings', 'lobyte')); cb.emit(0)
    cb.emit(0x85, 0x50)
    cb.emit(0xA5, 0x51)
    cb.emit(0x69); cb.patches.append((cb.foff(), 'decode_strings', 'hibyte')); cb.emit(0)
    cb.emit(0x85, 0x51)
    cb.emit(0xA0, 0x00)
    cb.label('_pt_lp')
    cb.emit(0xB1, 0x50)
    cb.emit_branch(0xF0, '_pt_done')
    cb.emit(0x20, CHROUT & 0xFF, (CHROUT >> 8) & 0xFF)
    cb.emit(0xC8)
    cb.emit_jmp('_pt_lp')
    cb.label('_pt_done')
    cb.emit(0x60)

    # blip: short SID beep on voice 1. Frequency nudged by GPOS so each token
    # sounds slightly higher than the last — audible "filling up" feeling.
    # Preserves A, X, Y.
    cb.label('blip')
    cb.emit(0x48)              # PHA
    cb.emit(0x8A, 0x48)        # TXA; PHA
    cb.emit(0x98, 0x48)        # TYA; PHA
    # Volume = 15
    cb.emit(0xA9, 0x0F)
    cb.emit(0x8D, 0x18, 0xD4)  # STA $D418
    # Attack=0, Decay=9 (quick decay)
    cb.emit(0xA9, 0x09)
    cb.emit(0x8D, 0x05, 0xD4)  # STA $D405
    # Sustain=0, Release=0
    cb.emit(0xA9, 0x00)
    cb.emit(0x8D, 0x06, 0xD4)  # STA $D406
    # Freq lo = $00, freq hi = $10 + GPOS*2
    cb.emit(0xA9, 0x00)
    cb.emit(0x8D, 0x00, 0xD4)  # STA $D400
    cb.emit(0xAD, GPOS & 0xFF, (GPOS >> 8) & 0xFF)
    cb.emit(0x0A)              # ASL (GPOS * 2)
    cb.emit(0x18, 0x69, 0x10)  # CLC; ADC #$10  (base freq hi)
    cb.emit(0x8D, 0x01, 0xD4)  # STA $D401
    # Gate on — triangle waveform ($10) + gate bit ($01) = $11
    cb.emit(0xA9, 0x11)
    cb.emit(0x8D, 0x04, 0xD4)  # STA $D404
    # Short delay loop (~20ms)
    cb.emit(0xA2, 0x40)        # LDX #$40
    cb.label('_blip_d1')
    cb.emit(0xA0, 0xFF)        # LDY #$FF
    cb.label('_blip_d2')
    cb.emit(0x88)              # DEY
    cb.emit(0xD0, 0xFD)        # BNE _blip_d2
    cb.emit(0xCA)              # DEX
    cb.emit(0xD0, 0xF8)        # BNE _blip_d1
    # Gate off — triangle only, no gate = $10
    cb.emit(0xA9, 0x10)
    cb.emit(0x8D, 0x04, 0xD4)  # STA $D404
    cb.emit(0x68, 0xA8)        # PLA; TAY
    cb.emit(0x68, 0xAA)        # PLA; TAX
    cb.emit(0x68)              # PLA
    cb.emit(0x60)              # RTS

    # ═══ Core inference routines ═══
    # build_rms_norm includes smul16 + isqrt32 + udiv as helpers
    build_rms_norm(cb)
    # sdiv used by attn_head
    build_sdiv_i32_u16(cb)
    # matvec + matvec_bias
    build_matvec(cb)
    # attn_head (references smul16 + sdiv already built)
    build_attn_head(cb)
    # embed + argmax
    build_embed_one_v2(cb)
    build_argmax(cb)

    # ═══ run_inference: autoregressive generation ═══
    cb.label('run_inference')
    cb.emit(0xA9, 0x0D); cb.emit(0x20, CHROUT & 0xFF, (CHROUT >> 8) & 0xFF)
    cb.emit(0xA9); cb.patches.append((cb.foff(), 'bob_str', 'lobyte')); cb.emit(0)
    cb.emit(0xA0); cb.patches.append((cb.foff(), 'bob_str', 'hibyte')); cb.emit(0)
    cb.emit_jsr('print_str')
    cb.emit(0xA9, 0x00)
    cb.emit(0x8D, GPOS & 0xFF, (GPOS >> 8) & 0xFF)

    cb.label('_gen_loop')
    cb.emit_jsr('do_forward')
    # A = argmax token
    # Check stop
    cb.emit(0xC9, END_TOK); cb.emit_branch_far(0xF0, '_gen_done')
    cb.emit(0xC9, SEP_TOK); cb.emit_branch_far(0xF0, '_gen_done')
    cb.emit(0xC9, PAD_TOK); cb.emit_branch_far(0xF0, '_gen_done')
    # Append token to TOKS
    cb.emit(0x48)  # PHA
    cb.emit(0xAE, SLEN & 0xFF, (SLEN >> 8) & 0xFF)
    cb.emit(0x9D, TOKS & 0xFF, (TOKS >> 8) & 0xFF)
    cb.emit(0xEE, SLEN & 0xFF, (SLEN >> 8) & 0xFF)
    cb.emit(0x68)  # PLA
    cb.emit_jsr('print_token')
    cb.emit_jsr('blip')
    # Check generation limit
    cb.emit(0xEE, GPOS & 0xFF, (GPOS >> 8) & 0xFF)
    cb.emit(0xAD, GPOS & 0xFF, (GPOS >> 8) & 0xFF)
    cb.emit(0xC9, 20)
    cb.emit_branch_far(0xB0, '_gen_done')
    cb.emit(0xAD, SLEN & 0xFF, (SLEN >> 8) & 0xFF)
    cb.emit(0xC9, MAXSEQ)
    cb.emit_branch_far(0xB0, '_gen_done')
    cb.emit_jmp('_gen_loop')
    cb.label('_gen_done')
    cb.emit(0x60)

    # ═══ do_forward: one forward pass, returns argmax in A ═══
    # Build weight address lookup. For each tensor we need to know its
    # absolute address in memory ($4000 + offset) and its shift.
    w_addrs = {}
    for name, kind, offset, size, shift in tensor_info:
        w_addrs[name] = (WEIGHTS_ADDR + offset, shift)

    cb.label('do_forward')
    cb.emit(0x78)  # SEI — disable interrupts during forward pass
    # ── 1. Embedding: for each position, embed_one ──
    cb.emit(0xA9, 0x00); cb.emit(0x85, 0x60)   # pos counter at $40

    cb.label('_fwd_emb_lp')
    # Set up TP = te_addr + TOKS[pos]*ED
    cb.emit(0xA5, 0x60)   # A = pos
    cb.emit(0xAA)          # X = pos
    cb.emit(0xBD, TOKS & 0xFF, (TOKS >> 8) & 0xFF)  # A = TOKS[pos]
    # Multiply A by ED (=32): A*32. Since A<128 and ED=32, result fits in 16 bits.
    # Use 5 left shifts.
    te_base = w_addrs['te'][0]
    cb.emit(0x85, 0x61)   # save tok id
    cb.emit(0xA9, 0x00); cb.emit(0x85, TP + 1)
    cb.emit(0xA5, 0x61)
    for _ in range(5):
        cb.emit(0x0A); cb.emit(0x26, TP + 1)   # ASL A; ROL TP+1
    cb.emit(0x18)
    cb.emit(0x69, te_base & 0xFF); cb.emit(0x85, TP)
    cb.emit(0xA5, TP + 1); cb.emit(0x69, (te_base >> 8) & 0xFF); cb.emit(0x85, TP + 1)

    # PP = pe_addr + pos*ED
    pe_base = w_addrs['pe'][0]
    cb.emit(0xA9, 0x00); cb.emit(0x85, PP + 1)
    cb.emit(0xA5, 0x60)   # pos
    for _ in range(5):
        cb.emit(0x0A); cb.emit(0x26, PP + 1)
    cb.emit(0x18)
    cb.emit(0x69, pe_base & 0xFF); cb.emit(0x85, PP)
    cb.emit(0xA5, PP + 1); cb.emit(0x69, (pe_base >> 8) & 0xFF); cb.emit(0x85, PP + 1)

    # DP = HIDDEN + pos*STRIDE
    cb.emit(0xA9, 0x00); cb.emit(0x85, DP + 1)
    cb.emit(0xA5, 0x60)
    for _ in range(6):
        cb.emit(0x0A); cb.emit(0x26, DP + 1)
    cb.emit(0x18)
    cb.emit(0x69, HIDDEN & 0xFF); cb.emit(0x85, DP)
    cb.emit(0xA5, DP + 1); cb.emit(0x69, (HIDDEN >> 8) & 0xFF); cb.emit(0x85, DP + 1)

    # SH1, SH2
    cb.emit(0xA9, w_addrs['te'][1]); cb.emit(0x85, SH1)
    cb.emit(0xA9, w_addrs['pe'][1]); cb.emit(0x85, SH2)

    cb.emit_jsr('embed_one')
    cb.emit(0xE6, 0x60)   # pos++
    cb.emit(0xA5, 0x60)
    cb.emit(0xCD, SLEN & 0xFF, (SLEN >> 8) & 0xFF)  # CMP abs SLEN
    cb.emit_branch_far(0x90, '_fwd_emb_lp')

    # ── 2. Layers ──
    for L in range(NL):
        lay_addrs = {k.split('.', 1)[1]: w_addrs[k]
                     for k in w_addrs if k.startswith(f'l{L}.')}

        # ── 2a. Per position: rms_norm(n1) → Q/K/V matvecs ──
        cb.emit(0xA9, 0x00); cb.emit(0x85, 0x60)  # pos = 0
        cb.label(f'_fwd_L{L}_qkv_lp')

        # XP = HIDDEN + pos * STRIDE
        cb.emit(0xA9, 0x00); cb.emit(0x85, RMS_XP + 1)
        cb.emit(0xA5, 0x60)
        for _ in range(6): cb.emit(0x0A); cb.emit(0x26, RMS_XP + 1)
        cb.emit(0x18)
        cb.emit(0x69, HIDDEN & 0xFF); cb.emit(0x85, RMS_XP)
        cb.emit(0xA5, RMS_XP + 1); cb.emit(0x69, (HIDDEN >> 8) & 0xFF); cb.emit(0x85, RMS_XP + 1)

        # GP = n1 weight addr, SG = n1 shift
        n1_addr, n1_shift = lay_addrs['n1']
        cb.emit(0xA9, n1_addr & 0xFF); cb.emit(0x85, RMS_GP)
        cb.emit(0xA9, (n1_addr >> 8) & 0xFF); cb.emit(0x85, RMS_GP + 1)
        cb.emit(0xA9, n1_shift); cb.emit(0x85, RMS_SG)

        # DP = XN_BUF
        cb.emit(0xA9, XN_BUF & 0xFF); cb.emit(0x85, RMS_DP)
        cb.emit(0xA9, (XN_BUF >> 8) & 0xFF); cb.emit(0x85, RMS_DP + 1)

        cb.emit(0xA5, 0x60); cb.emit(0x48)  # save pos
        cb.emit_jsr('rms_norm')
        cb.emit(0x68); cb.emit(0x85, 0x60)  # restore pos

        # Q matvec: WP=q_addr, SP=XN_BUF, DP=Q_ALL+pos*STRIDE, ROWS=ED, COLS=ED, SHIFT=s+1
        for target_name, target_base in [('q', Q_ALL), ('k', K_ALL), ('v', V_ALL)]:
            t_addr, t_shift = lay_addrs[target_name]
            cb.emit(0xA9, t_addr & 0xFF); cb.emit(0x85, WP)
            cb.emit(0xA9, (t_addr >> 8) & 0xFF); cb.emit(0x85, WP + 1)
            cb.emit(0xA9, XN_BUF & 0xFF); cb.emit(0x85, SP)
            cb.emit(0xA9, (XN_BUF >> 8) & 0xFF); cb.emit(0x85, SP + 1)
            # DP = target_base + pos * STRIDE
            cb.emit(0xA9, 0x00); cb.emit(0x85, DP + 1)
            cb.emit(0xA5, 0x60)
            for _ in range(6): cb.emit(0x0A); cb.emit(0x26, DP + 1)
            cb.emit(0x18); cb.emit(0x69, target_base & 0xFF); cb.emit(0x85, DP)
            cb.emit(0xA5, DP + 1); cb.emit(0x69, (target_base >> 8) & 0xFF); cb.emit(0x85, DP + 1)
            cb.emit(0xA9, ED); cb.emit(0x85, ROWS)
            cb.emit(0xA9, ED); cb.emit(0x85, COLS)
            cb.emit(0xA9, t_shift + 1); cb.emit(0x85, SHIFT)  # post_shift=1
            cb.emit(0xA5, 0x60); cb.emit(0x48)
            cb.emit_jsr('matvec')
            cb.emit(0x68); cb.emit(0x85, 0x60)

        # pos++
        cb.emit(0xE6, 0x60)
        cb.emit(0xA5, 0x60)
        cb.emit(0xCD, SLEN & 0xFF, (SLEN >> 8) & 0xFF)
        cb.emit_branch_far(0x90, f'_fwd_L{L}_qkv_lp')

        # ── 2b. Attention: per position, per head ──
        cb.emit(0xA9, 0x00); cb.emit(0x85, 0x60)  # pos = 0 (t_q)
        cb.label(f'_fwd_L{L}_att_pos_lp')
        # Heartbeat: rotate border color each attention position.
        # INC $D020 — 20 positions × 2 layers = 40 color changes per forward pass.
        cb.emit(0xEE, 0x20, 0xD0)
        cb.emit(0xA9, 0x00); cb.emit(0x85, 0x61)  # head = 0
        cb.label(f'_fwd_L{L}_att_head_lp')
        # Set up attn_head parameters
        # QP = Q_ALL + pos*STRIDE + head*HD*2
        cb.emit(0xA9, 0x00); cb.emit(0x85, QP + 1)
        cb.emit(0xA5, 0x60)
        for _ in range(6): cb.emit(0x0A); cb.emit(0x26, QP + 1)
        cb.emit(0x18); cb.emit(0x69, Q_ALL & 0xFF); cb.emit(0x85, QP)
        cb.emit(0xA5, QP + 1); cb.emit(0x69, (Q_ALL >> 8) & 0xFF); cb.emit(0x85, QP + 1)
        # Add head*16
        cb.emit(0xA5, 0x61)  # head
        cb.emit(0x0A); cb.emit(0x0A); cb.emit(0x0A); cb.emit(0x0A)  # *16
        cb.emit(0x18); cb.emit(0x65, QP); cb.emit(0x85, QP)
        cb.emit(0x90, 0x02); cb.emit(0xE6, QP + 1)

        # KB, VB
        cb.emit(0xA9, K_ALL & 0xFF); cb.emit(0x85, KB)
        cb.emit(0xA9, (K_ALL >> 8) & 0xFF); cb.emit(0x85, KB + 1)
        cb.emit(0xA9, V_ALL & 0xFF); cb.emit(0x85, VB)
        cb.emit(0xA9, (V_ALL >> 8) & 0xFF); cb.emit(0x85, VB + 1)

        # OP = ATT_OUT + pos*STRIDE + head*HD*2
        cb.emit(0xA9, 0x00); cb.emit(0x85, OP + 1)
        cb.emit(0xA5, 0x60)
        for _ in range(6): cb.emit(0x0A); cb.emit(0x26, OP + 1)
        cb.emit(0x18); cb.emit(0x69, ATT_OUT & 0xFF); cb.emit(0x85, OP)
        cb.emit(0xA5, OP + 1); cb.emit(0x69, (ATT_OUT >> 8) & 0xFF); cb.emit(0x85, OP + 1)
        cb.emit(0xA5, 0x61)
        cb.emit(0x0A); cb.emit(0x0A); cb.emit(0x0A); cb.emit(0x0A)
        cb.emit(0x18); cb.emit(0x65, OP); cb.emit(0x85, OP)
        cb.emit(0x90, 0x02); cb.emit(0xE6, OP + 1)

        # NKEYS = pos + 1 (causal: attend to 0..pos)
        cb.emit(0xA5, 0x60); cb.emit(0x18); cb.emit(0x69, 0x01)
        cb.emit(0x85, NKEYS)
        cb.emit(0xA5, 0x61); cb.emit(0x85, HEAD)

        # Scratch buffers
        cb.emit(0xA9, SCORES_BUF & 0xFF); cb.emit(0x85, SCORES_P)
        cb.emit(0xA9, (SCORES_BUF >> 8) & 0xFF); cb.emit(0x85, SCORES_P + 1)
        cb.emit(0xA9, WEIGHTS_BUF & 0xFF); cb.emit(0x85, WTS_P)
        cb.emit(0xA9, (WEIGHTS_BUF >> 8) & 0xFF); cb.emit(0x85, WTS_P + 1)
        cb.emit(0xA9); cb.patches.append((cb.foff(), 'exp_lut', 'lobyte')); cb.emit(0)
        cb.emit(0x85, ELUT_P)
        cb.emit(0xA9); cb.patches.append((cb.foff(), 'exp_lut', 'hibyte')); cb.emit(0)
        cb.emit(0x85, ELUT_P + 1)

        # Save pos and head, call attn_head
        cb.emit(0xA5, 0x60); cb.emit(0x48)
        cb.emit(0xA5, 0x61); cb.emit(0x48)
        cb.emit_jsr('attn_head')
        cb.emit(0x68); cb.emit(0x85, 0x61)
        cb.emit(0x68); cb.emit(0x85, 0x60)

        # Next head
        cb.emit(0xE6, 0x61)
        cb.emit(0xA5, 0x61); cb.emit(0xC9, NH)
        cb.emit_branch_far(0x90, f'_fwd_L{L}_att_head_lp')

        # Next pos
        cb.emit(0xE6, 0x60)
        cb.emit(0xA5, 0x60)
        cb.emit(0xCD, SLEN & 0xFF, (SLEN >> 8) & 0xFF)
        cb.emit_branch_far(0x90, f'_fwd_L{L}_att_pos_lp')

        # ── 2c. Proj + residual per position ──
        proj_addr, proj_shift = lay_addrs['proj']
        cb.emit(0xA9, 0x00); cb.emit(0x85, 0x60)
        cb.label(f'_fwd_L{L}_proj_lp')
        # WP = proj addr
        cb.emit(0xA9, proj_addr & 0xFF); cb.emit(0x85, WP)
        cb.emit(0xA9, (proj_addr >> 8) & 0xFF); cb.emit(0x85, WP + 1)
        # SP = ATT_OUT + pos*STRIDE
        cb.emit(0xA9, 0x00); cb.emit(0x85, SP + 1)
        cb.emit(0xA5, 0x60)
        for _ in range(6): cb.emit(0x0A); cb.emit(0x26, SP + 1)
        cb.emit(0x18); cb.emit(0x69, ATT_OUT & 0xFF); cb.emit(0x85, SP)
        cb.emit(0xA5, SP + 1); cb.emit(0x69, (ATT_OUT >> 8) & 0xFF); cb.emit(0x85, SP + 1)
        # DP = W2_BUF (scratch)
        cb.emit(0xA9, W2_BUF & 0xFF); cb.emit(0x85, DP)
        cb.emit(0xA9, (W2_BUF >> 8) & 0xFF); cb.emit(0x85, DP + 1)
        cb.emit(0xA9, ED); cb.emit(0x85, ROWS); cb.emit(0x85, COLS)
        cb.emit(0xA9, proj_shift + 1); cb.emit(0x85, SHIFT)
        cb.emit(0xA5, 0x60); cb.emit(0x48)
        cb.emit_jsr('matvec')
        cb.emit(0x68); cb.emit(0x85, 0x60)
        # Residual: HIDDEN[pos] += W2_BUF
        # DP = HIDDEN + pos*STRIDE, SP2 = W2_BUF
        cb.emit(0xA9, 0x00); cb.emit(0x85, DP + 1)
        cb.emit(0xA5, 0x60)
        for _ in range(6): cb.emit(0x0A); cb.emit(0x26, DP + 1)
        cb.emit(0x18); cb.emit(0x69, HIDDEN & 0xFF); cb.emit(0x85, DP)
        cb.emit(0xA5, DP + 1); cb.emit(0x69, (HIDDEN >> 8) & 0xFF); cb.emit(0x85, DP + 1)
        cb.emit(0xA9, W2_BUF & 0xFF); cb.emit(0x85, SP)  # SP2 for residual_add
        cb.emit(0xA9, (W2_BUF >> 8) & 0xFF); cb.emit(0x85, SP + 1)
        cb.emit(0xA5, 0x60); cb.emit(0x48)
        cb.emit_jsr('residual_add')
        cb.emit(0x68); cb.emit(0x85, 0x60)
        cb.emit(0xE6, 0x60)
        cb.emit(0xA5, 0x60)
        cb.emit(0xCD, SLEN & 0xFF, (SLEN >> 8) & 0xFF)
        cb.emit_branch_far(0x90, f'_fwd_L{L}_proj_lp')

        # ── 2e. FFN per position ──
        n2_addr, n2_shift = lay_addrs['n2']
        fc1w_addr, fc1w_shift = lay_addrs['fc1_w']
        fc1b_addr, fc1b_shift = lay_addrs['fc1_b']
        fc2w_addr, fc2w_shift = lay_addrs['fc2_w']
        fc2b_addr, fc2b_shift = lay_addrs['fc2_b']

        cb.emit(0xA9, 0x00); cb.emit(0x85, 0x60)
        cb.label(f'_fwd_L{L}_ffn_lp')
        # rms_norm(h[pos], n2) → XN_BUF
        cb.emit(0xA9, 0x00); cb.emit(0x85, RMS_XP + 1)
        cb.emit(0xA5, 0x60)
        for _ in range(6): cb.emit(0x0A); cb.emit(0x26, RMS_XP + 1)
        cb.emit(0x18); cb.emit(0x69, HIDDEN & 0xFF); cb.emit(0x85, RMS_XP)
        cb.emit(0xA5, RMS_XP + 1); cb.emit(0x69, (HIDDEN >> 8) & 0xFF); cb.emit(0x85, RMS_XP + 1)
        cb.emit(0xA9, n2_addr & 0xFF); cb.emit(0x85, RMS_GP)
        cb.emit(0xA9, (n2_addr >> 8) & 0xFF); cb.emit(0x85, RMS_GP + 1)
        cb.emit(0xA9, n2_shift); cb.emit(0x85, RMS_SG)
        cb.emit(0xA9, XN_BUF & 0xFF); cb.emit(0x85, RMS_DP)
        cb.emit(0xA9, (XN_BUF >> 8) & 0xFF); cb.emit(0x85, RMS_DP + 1)
        cb.emit(0xA5, 0x60); cb.emit(0x48)
        cb.emit_jsr('rms_norm')
        cb.emit(0x68); cb.emit(0x85, 0x60)

        # fc1: matvec_bias(fc1_w, fc1_b, XN_BUF) → Z_BUF
        cb.emit(0xA9, fc1w_addr & 0xFF); cb.emit(0x85, WP)
        cb.emit(0xA9, (fc1w_addr >> 8) & 0xFF); cb.emit(0x85, WP + 1)
        cb.emit(0xA9, XN_BUF & 0xFF); cb.emit(0x85, SP)
        cb.emit(0xA9, (XN_BUF >> 8) & 0xFF); cb.emit(0x85, SP + 1)
        cb.emit(0xA9, Z_BUF & 0xFF); cb.emit(0x85, DP)
        cb.emit(0xA9, (Z_BUF >> 8) & 0xFF); cb.emit(0x85, DP + 1)
        cb.emit(0xA9, fc1b_addr & 0xFF); cb.emit(0x85, BP)
        cb.emit(0xA9, (fc1b_addr >> 8) & 0xFF); cb.emit(0x85, BP + 1)
        cb.emit(0xA9, FF); cb.emit(0x85, ROWS)
        cb.emit(0xA9, ED); cb.emit(0x85, COLS)
        cb.emit(0xA9, fc1w_shift + 1); cb.emit(0x85, SHIFT)
        cb.emit(0xA5, 0x60); cb.emit(0x48)
        cb.emit_jsr('matvec_bias')
        cb.emit(0x68); cb.emit(0x85, 0x60)

        # ReLU on Z_BUF
        cb.emit(0xA9, Z_BUF & 0xFF); cb.emit(0x85, DP)
        cb.emit(0xA9, (Z_BUF >> 8) & 0xFF); cb.emit(0x85, DP + 1)
        cb.emit(0xA9, FF); cb.emit(0x85, 0xF6)  # ROWS for relu
        cb.emit_jsr('relu')

        # fc2: matvec_bias(fc2_w, fc2_b, Z_BUF) → W2_BUF
        cb.emit(0xA9, fc2w_addr & 0xFF); cb.emit(0x85, WP)
        cb.emit(0xA9, (fc2w_addr >> 8) & 0xFF); cb.emit(0x85, WP + 1)
        cb.emit(0xA9, Z_BUF & 0xFF); cb.emit(0x85, SP)
        cb.emit(0xA9, (Z_BUF >> 8) & 0xFF); cb.emit(0x85, SP + 1)
        cb.emit(0xA9, W2_BUF & 0xFF); cb.emit(0x85, DP)
        cb.emit(0xA9, (W2_BUF >> 8) & 0xFF); cb.emit(0x85, DP + 1)
        cb.emit(0xA9, fc2b_addr & 0xFF); cb.emit(0x85, BP)
        cb.emit(0xA9, (fc2b_addr >> 8) & 0xFF); cb.emit(0x85, BP + 1)
        cb.emit(0xA9, ED); cb.emit(0x85, ROWS)
        cb.emit(0xA9, FF); cb.emit(0x85, COLS)
        cb.emit(0xA9, fc2w_shift + 1); cb.emit(0x85, SHIFT)
        cb.emit(0xA5, 0x60); cb.emit(0x48)
        cb.emit_jsr('matvec_bias')
        cb.emit(0x68); cb.emit(0x85, 0x60)

        # Residual: HIDDEN[pos] += W2_BUF
        cb.emit(0xA9, 0x00); cb.emit(0x85, DP + 1)
        cb.emit(0xA5, 0x60)
        for _ in range(6): cb.emit(0x0A); cb.emit(0x26, DP + 1)
        cb.emit(0x18); cb.emit(0x69, HIDDEN & 0xFF); cb.emit(0x85, DP)
        cb.emit(0xA5, DP + 1); cb.emit(0x69, (HIDDEN >> 8) & 0xFF); cb.emit(0x85, DP + 1)
        cb.emit(0xA9, W2_BUF & 0xFF); cb.emit(0x85, SP)
        cb.emit(0xA9, (W2_BUF >> 8) & 0xFF); cb.emit(0x85, SP + 1)
        cb.emit(0xA5, 0x60); cb.emit(0x48)
        cb.emit_jsr('residual_add')
        cb.emit(0x68); cb.emit(0x85, 0x60)

        cb.emit(0xE6, 0x60)
        cb.emit(0xA5, 0x60)
        cb.emit(0xCD, SLEN & 0xFF, (SLEN >> 8) & 0xFF)
        cb.emit_branch_far(0x90, f'_fwd_L{L}_ffn_lp')

    # ── 3. Final norm + output projection ──
    norm_addr, norm_shift = w_addrs['norm']
    out_addr, out_shift = w_addrs['out']

    # rms_norm(HIDDEN[last], norm) → XN_BUF
    # XP = HIDDEN + (SLEN-1)*STRIDE
    cb.emit(0xAD, SLEN & 0xFF, (SLEN >> 8) & 0xFF)
    cb.emit(0x38); cb.emit(0xE9, 0x01)   # A = SLEN - 1
    cb.emit(0x85, 0x60)
    cb.emit(0xA9, 0x00); cb.emit(0x85, RMS_XP + 1)
    cb.emit(0xA5, 0x60)
    for _ in range(6): cb.emit(0x0A); cb.emit(0x26, RMS_XP + 1)
    cb.emit(0x18); cb.emit(0x69, HIDDEN & 0xFF); cb.emit(0x85, RMS_XP)
    cb.emit(0xA5, RMS_XP + 1); cb.emit(0x69, (HIDDEN >> 8) & 0xFF); cb.emit(0x85, RMS_XP + 1)
    cb.emit(0xA9, norm_addr & 0xFF); cb.emit(0x85, RMS_GP)
    cb.emit(0xA9, (norm_addr >> 8) & 0xFF); cb.emit(0x85, RMS_GP + 1)
    cb.emit(0xA9, norm_shift); cb.emit(0x85, RMS_SG)
    cb.emit(0xA9, XN_BUF & 0xFF); cb.emit(0x85, RMS_DP)
    cb.emit(0xA9, (XN_BUF >> 8) & 0xFF); cb.emit(0x85, RMS_DP + 1)
    cb.emit_jsr('rms_norm')

    # Output matvec: WP=out, SP=XN_BUF, DP=LOGITS_BUF, ROWS=VS, COLS=ED, SHIFT=out_shift (no +1!)
    # LOGITS_BUF must NOT overlap XN_BUF (we read XN_BUF while writing LOGITS),
    # and MUST be in always-RAM (not screen, not BASIC ROM).
    LOGITS_BUF = 0xC100
    cb.emit(0xA9, out_addr & 0xFF); cb.emit(0x85, WP)
    cb.emit(0xA9, (out_addr >> 8) & 0xFF); cb.emit(0x85, WP + 1)
    cb.emit(0xA9, XN_BUF & 0xFF); cb.emit(0x85, SP)
    cb.emit(0xA9, (XN_BUF >> 8) & 0xFF); cb.emit(0x85, SP + 1)
    cb.emit(0xA9, LOGITS_BUF & 0xFF); cb.emit(0x85, DP)
    cb.emit(0xA9, (LOGITS_BUF >> 8) & 0xFF); cb.emit(0x85, DP + 1)
    cb.emit(0xA9, VS); cb.emit(0x85, ROWS)
    cb.emit(0xA9, ED); cb.emit(0x85, COLS)
    cb.emit(0xA9, out_shift); cb.emit(0x85, SHIFT)   # post_shift=0
    cb.emit_jsr('matvec')

    # argmax
    cb.emit(0xA9, LOGITS_BUF & 0xFF); cb.emit(0x85, DP)
    cb.emit(0xA9, (LOGITS_BUF >> 8) & 0xFF); cb.emit(0x85, DP + 1)
    cb.emit_jsr('argmax')
    # A = token id. Reset border to green, preserving A.
    cb.emit(0x48)                     # PHA
    cb.emit(0xA9, 0x05)               # LDA #5
    cb.emit(0x8D, 0x20, 0xD0)         # STA $D020
    cb.emit(0x68)                     # PLA
    cb.emit(0x58)                     # CLI — re-enable interrupts
    cb.emit(0x60)   # RTS

    # ═══ residual_add (inline from asm_simple) ═══
    from asm_simple import build_residual_add, build_relu
    build_residual_add(cb)
    build_relu(cb)

    # ═══ Data section ═══
    cb.label('banner')
    cb.emit_str("\r  .--------.\r  | O    O  |\r  |    V    |\r  |..|---|..|\r\rSOUL PLAYER C64\r2026-04-06 - GIZMO64K\r\r25k PARAMETERS\rREAL TRANSFORMER. REAL WEIGHTS.\rLOADED OFF A FLOPPY DISK.\r")
    cb.label('ready_msg')
    cb.emit_str("\r2 LAYERS, 4 HEADS, REAL ATTENTION,\rINT8, 64K!\r\rTYPE AND I WILL TALK BACK\rEVENTUALLY... AFTER MINUTES!\rTYPE 'Q' TO QUIT.\r")
    cb.label('prompt_str')
    cb.emit_str("YOU> ")
    cb.label('bob_str')
    cb.emit_str("C64> ")
    cb.label('quit_msg')
    cb.emit_str("\r-- ATTENTION IS ALL THIS NEEDED\rGIZMO64K\r")

    # EXP_LUT
    cb.label('exp_lut')
    cb.emit_data(bytes(int(v) for v in EXP_LUT.tolist()))

    # Tokenizer tables
    cb.label('decode_offsets')
    cb.emit_data(tok_offsets)
    cb.label('decode_strings')
    cb.emit_data(tok_strings)
    cb.label('merge_table')
    cb.emit_data(tok_merges)

    print(f"  Code+data: {len(cb.buf)} bytes (${cb.org:04X}-${cb.pc:04X})")
    if cb.pc >= WEIGHTS_ADDR:
        print(f"  WARNING: code overlaps weight region at ${WEIGHTS_ADDR:04X}")

    return cb


# ─── D64 builder ──────────────────────────────────────────────────
def build_d64_single(prg_data):
    SPT = [0] + [21] * 17 + [19] * 7 + [18] * 6 + [17] * 5
    TOTAL = sum(SPT[1:])
    disk = bytearray(TOTAL * 256)
    def ts2off(t, s):
        o = 0
        for i in range(1, t): o += SPT[i] * 256
        return o + s * 256
    def write_sec(t, s, data):
        o = ts2off(t, s)
        for i in range(min(256, len(data))): disk[o + i] = data[i]
    def alloc(t, s):
        o = ts2off(18, 0); b = 4 + (t - 1) * 4
        disk[o + b] = max(0, disk[o + b] - 1)
        disk[o + b + 1 + s // 8] &= ~(1 << (s % 8))
    bam = bytearray(256)
    bam[0] = 18; bam[1] = 1; bam[2] = 0x41
    for t in range(1, 36):
        b = 4 + (t - 1) * 4; n = SPT[t]; bam[b] = n; bits = (1 << n) - 1
        bam[b + 1] = bits & 0xFF; bam[b + 2] = (bits >> 8) & 0xFF; bam[b + 3] = (bits >> 16) & 0xFF
    for i in range(16): bam[0x90 + i] = 0xA0
    for i, c in enumerate("SOUL PLAYER"): bam[0x90 + i] = ord(c)
    bam[0xA2] = ord('B'); bam[0xA3] = ord('F'); bam[0xA4] = 0xA0
    bam[0xA5] = ord('2'); bam[0xA6] = ord('A')
    for i in range(0xA7, 0xAB): bam[i] = 0xA0
    write_sec(18, 0, bam); alloc(18, 0)
    class Al:
        def __init__(self): self.t = 1; self.s = 0
        def next(self):
            t, s = self.t, self.s; self.s += 1
            if self.s >= SPT[self.t]: self.s = 0; self.t += 1
            if self.t == 18: self.t = 19
            return t, s
    al = Al()
    ns = (len(prg_data) + 253) // 254
    secs = [al.next() for _ in range(ns)]
    for i, (t, s) in enumerate(secs):
        alloc(t, s)
        blk = bytearray(256)
        chunk = prg_data[i * 254:(i + 1) * 254]
        if i < ns - 1: blk[0], blk[1] = secs[i + 1]
        else: blk[0] = 0; blk[1] = len(chunk) + 1
        for j, b in enumerate(chunk): blk[2 + j] = b
        write_sec(t, s, blk)
    alloc(18, 1)
    d = bytearray(256); d[0] = 0; d[1] = 0xFF
    e = 2; d[e] = 0x82; d[e + 1] = secs[0][0]; d[e + 2] = secs[0][1]
    for i in range(16): d[e + 3 + i] = 0xA0
    for i, c in enumerate("BOB"): d[e + 3 + i] = ord(c)
    d[e + 0x1C] = len(secs) & 0xFF; d[e + 0x1D] = (len(secs) >> 8) & 0xFF
    write_sec(18, 1, d)
    return bytes(disk)


# ─── Main ──────────────────────────────────────────────────────────
def main():
    import argparse
    from pathlib import Path

    # Resolve repo root: this file is src/build.py, so root is one level up
    REPO_ROOT = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Soul C64 Builder — assemble .prg and .d64")
    parser.add_argument("--soul", default=str(REPO_ROOT / "models" / "soul.bin"),
                        help="Path to trained soul .bin file (default: models/soul.bin)")
    parser.add_argument("--tokenizer", default=str(REPO_ROOT / "models" / "tokenizer.json"),
                        help="Path to tokenizer .json file (default: models/tokenizer.json)")
    parser.add_argument("--output", default=str(REPO_ROOT / "disk"),
                        help="Output directory for .prg and .d64 (default: disk/)")
    args = parser.parse_args()

    soul_path = args.soul
    tok_path = args.tokenizer
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  SOUL C64 BUILDER — REAL TRANSFORMER")
    print("=" * 55)

    if not os.path.exists(soul_path):
        print(f"\n  ERROR: {soul_path} not found.")
        print(f"  Train a model first:  python train.py data/corpus.txt")
        sys.exit(1)
    if not os.path.exists(tok_path):
        print(f"\n  ERROR: {tok_path} not found.")
        print(f"  Train a model first:  python train.py data/corpus.txt")
        sys.exit(1)

    print(f"\n  Soul:      {soul_path}")
    print(f"  Tokenizer: {tok_path}")
    print(f"  Output:    {out_dir}/")

    print("\n--- Soul ---")
    soul_blob, tensor_info = parse_soul_for_c64(soul_path)
    print(f"  {len(soul_blob)} bytes raw weights")
    for name, kind, offset, size, shift in tensor_info:
        print(f"    {name:15s} {kind} off={offset:5d} size={size:5d} shift={shift:+d}")

    print("\n--- Tokenizer ---")
    tok_off, tok_str, tok_merge = build_tokenizer_tables(tok_path)
    print(f"  {len(tok_off)}B offsets, {len(tok_str)}B strings, {len(tok_merge)}B merges")

    print("\n--- PRG ---")
    cb = build_program(soul_blob, tensor_info, tok_off, tok_str, tok_merge)
    prg = cb.get_prg()

    # Pad from code end to WEIGHTS_ADDR, then append weights
    code_end = 0x0801 + len(prg) - 2
    pad = WEIGHTS_ADDR - code_end
    if pad < 0:
        print(f"  ERROR: code extends past weights! Code ends at ${code_end:04X}")
        return
    print(f"  Code ends at ${code_end:04X}, padding {pad} bytes to ${WEIGHTS_ADDR:04X}")
    combined = bytearray(prg) + b'\x00' * pad + soul_blob
    w_end = WEIGHTS_ADDR + len(soul_blob)
    print(f"  Weights: ${WEIGHTS_ADDR:04X}-${w_end:04X} ({len(soul_blob)} bytes)")
    print(f"  Combined PRG: {len(combined)} bytes ({len(combined)/1024:.1f}KB)")

    print("\n--- D64 ---")
    d64 = build_d64_single(combined)
    print(f"  D64: {len(d64)} bytes")

    prg_path = out_dir / "soulplayer.prg"
    d64_path = out_dir / "soulplayer.d64"
    open(prg_path, 'wb').write(bytes(combined))
    open(d64_path, 'wb').write(d64)
    print(f"\n  {prg_path}:  {len(combined)} bytes")
    print(f"  {d64_path}:  {len(d64)} bytes")
    print(f"\n  LOAD\"SOULPLAYER\",8,1")
    print(f"  RUN")


if __name__ == '__main__':
    main()
