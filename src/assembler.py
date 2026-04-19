#!/usr/bin/env python3
"""
Shared test harness for 6502 routines.

Each routine under test:
  1. Is built by a `build_*` function that appends its assembly to a
     CodeBuilder, labels its entry point, and optionally labels any
     helpers or data it needs.
  2. Gets called via a test wrapper that creates a fresh CPU, loads the
     code, sets up zero-page pointers + memory state, calls the routine
     via JSR, BRK after return, and reads back the result.
  3. Is compared byte-for-byte against its Python shadow counterpart
     (`op_*` in bob_shadow_v3.py) on the same inputs.

A routine is considered correct only when all test cases match exactly.
"""
import sys
from cpu6502 import CPU


class CodeBuilder:
    """Compact 6502 assembler with labels, patches, and far-branch helper.
    Copied/simplified from the v3/v4 builders."""

    def __init__(self, org=0x0800):
        self.org = org
        self.buf = bytearray()
        self.labels = {}
        self.patches = []  # (file_offset, label, type)

    @property
    def pc(self):
        return self.org + len(self.buf)

    def foff(self):
        return len(self.buf)

    def emit(self, *bs):
        for b in bs:
            self.buf.append(b & 0xFF)

    def label(self, name):
        if name in self.labels:
            raise ValueError(f"duplicate label: {name}")
        self.labels[name] = self.pc

    def emit_jsr(self, lbl):
        self.emit(0x20)
        self.patches.append((self.foff(), lbl, 'abs16'))
        self.emit(0x00, 0x00)

    def emit_jmp(self, lbl):
        self.emit(0x4C)
        self.patches.append((self.foff(), lbl, 'abs16'))
        self.emit(0x00, 0x00)

    def emit_branch(self, opcode, lbl):
        """Short relative branch — fails at link time if distance > rel8."""
        self.emit(opcode)
        self.patches.append((self.foff(), lbl, 'rel8'))
        self.emit(0x00)

    def emit_branch_far(self, opcode, lbl):
        """Conditional branch that works at any distance via inverted-branch + JMP."""
        invert = {0x90: 0xB0, 0xB0: 0x90,
                  0xF0: 0xD0, 0xD0: 0xF0,
                  0x10: 0x30, 0x30: 0x10}
        inv = invert[opcode]
        self.emit(inv, 0x03)
        self.emit_jmp(lbl)

    def emit_data(self, data):
        for b in data:
            self.emit(b & 0xFF)

    def emit_str(self, s):
        """Emit a null-terminated PETSCII string."""
        for ch in s:
            if ch in '\r\n':
                self.emit(0x0D)
            elif 'A' <= ch <= 'Z':
                self.emit(ord(ch))
            elif 'a' <= ch <= 'z':
                self.emit(ord(ch) - 32)
            elif ch == ' ':
                self.emit(0x20)
            else:
                self.emit(ord(ch) & 0x7F)
        self.emit(0x00)

    def get_prg(self):
        """Return complete PRG file bytes (with 2-byte load address)."""
        self.resolve()
        import struct
        return struct.pack('<H', self.org) + bytes(self.buf)

    def resolve(self):
        for foff, lbl, ptype in self.patches:
            if lbl not in self.labels:
                raise ValueError(f"unresolved label: {lbl}")
            addr = self.labels[lbl]
            if ptype == 'abs16':
                self.buf[foff]     = addr & 0xFF
                self.buf[foff + 1] = (addr >> 8) & 0xFF
            elif ptype == 'lobyte':
                self.buf[foff] = addr & 0xFF
            elif ptype == 'hibyte':
                self.buf[foff] = (addr >> 8) & 0xFF
            elif ptype == 'rel8':
                offset = addr - (self.org + foff + 1)
                if offset < -128 or offset > 127:
                    raise ValueError(f"branch overflow to {lbl}: {offset}")
                self.buf[foff] = offset & 0xFF

    def get_bytes(self):
        self.resolve()
        return bytes(self.buf)


# ─── Convenient mnemonics ──────────────────────────────────────────
# Just a few of the ones we'll use a lot. Rest we'll emit as raw bytes
# with comments — clearer than inventing a full assembler.
def CLC(cb): cb.emit(0x18)
def SEC(cb): cb.emit(0x38)
def RTS(cb): cb.emit(0x60)
def BRK(cb): cb.emit(0x00)
def PHA(cb): cb.emit(0x48)
def PLA(cb): cb.emit(0x68)
def TXA(cb): cb.emit(0x8A)
def TAX(cb): cb.emit(0xAA)
def TYA(cb): cb.emit(0x98)
def TAY(cb): cb.emit(0xA8)
def INX(cb): cb.emit(0xE8)
def DEX(cb): cb.emit(0xCA)
def INY(cb): cb.emit(0xC8)
def DEY(cb): cb.emit(0x88)

def LDA_imm(cb, v):     cb.emit(0xA9, v & 0xFF)
def LDX_imm(cb, v):     cb.emit(0xA2, v & 0xFF)
def LDY_imm(cb, v):     cb.emit(0xA0, v & 0xFF)
def LDA_zp(cb, zp):     cb.emit(0xA5, zp)
def LDX_zp(cb, zp):     cb.emit(0xA6, zp)
def LDY_zp(cb, zp):     cb.emit(0xA4, zp)
def STA_zp(cb, zp):     cb.emit(0x85, zp)
def STX_zp(cb, zp):     cb.emit(0x86, zp)
def STY_zp(cb, zp):     cb.emit(0x84, zp)
def LDA_abs(cb, a):     cb.emit(0xAD, a & 0xFF, (a >> 8) & 0xFF)
def STA_abs(cb, a):     cb.emit(0x8D, a & 0xFF, (a >> 8) & 0xFF)
def LDA_indY(cb, zp):   cb.emit(0xB1, zp)
def STA_indY(cb, zp):   cb.emit(0x91, zp)
def LDA_absY(cb, a):    cb.emit(0xB9, a & 0xFF, (a >> 8) & 0xFF)
def LDA_absX(cb, a):    cb.emit(0xBD, a & 0xFF, (a >> 8) & 0xFF)
def ADC_imm(cb, v):     cb.emit(0x69, v & 0xFF)
def ADC_zp(cb, zp):     cb.emit(0x65, zp)
def SBC_imm(cb, v):     cb.emit(0xE9, v & 0xFF)
def SBC_zp(cb, zp):     cb.emit(0xE5, zp)
def AND_imm(cb, v):     cb.emit(0x29, v & 0xFF)
def EOR_imm(cb, v):     cb.emit(0x49, v & 0xFF)
def CMP_imm(cb, v):     cb.emit(0xC9, v & 0xFF)
def CPX_imm(cb, v):     cb.emit(0xE0, v & 0xFF)
def CPY_imm(cb, v):     cb.emit(0xC0, v & 0xFF)
def INC_zp(cb, zp):     cb.emit(0xE6, zp)
def DEC_zp(cb, zp):     cb.emit(0xC6, zp)
def ASL_zp(cb, zp):     cb.emit(0x06, zp)
def ROL_zp(cb, zp):     cb.emit(0x26, zp)
def LSR_zp(cb, zp):     cb.emit(0x46, zp)
def ROR_zp(cb, zp):     cb.emit(0x66, zp)
def ASL_A(cb):          cb.emit(0x0A)
def LSR_A(cb):          cb.emit(0x4A)


def run_subroutine(code_bytes, code_org, entry_label_addr, setup_cpu_fn,
                   max_cycles=30_000_000):
    """
    Run an assembled routine in isolation.

    code_bytes: the assembled bytes
    code_org:   address where code_bytes starts
    entry_label_addr: absolute address of the subroutine entry point
    setup_cpu_fn: callback (cpu) -> None, for setting zero-page/memory
    Returns the finished CPU object.
    """
    cpu = CPU()
    # Load code
    for i, b in enumerate(code_bytes):
        cpu.mem[code_org + i] = b
    # Install a tiny shim that JSRs the entry and then BRKs:
    # We place the shim at 0x0300 (away from typical working memory).
    shim = code_org - 16  # just before code
    cpu.mem[shim + 0] = 0x20  # JSR
    cpu.mem[shim + 1] = entry_label_addr & 0xFF
    cpu.mem[shim + 2] = (entry_label_addr >> 8) & 0xFF
    cpu.mem[shim + 3] = 0x00  # BRK
    setup_cpu_fn(cpu)
    cpu.pc = shim
    cpu.run(max_cycles=max_cycles)
    return cpu
