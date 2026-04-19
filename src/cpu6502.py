#!/usr/bin/env python3
"""
Minimal 6502 interpreter.

Implements just enough of the NMOS 6502 to run the bob inference routines.
Not cycle-accurate, not decimal-mode, no IRQs — just the opcodes we need,
with correct flag semantics for signed/unsigned comparisons and arithmetic.

Covered (all addressing modes used by the builder):
  Loads/stores:  LDA LDX LDY STA STX STY
  Transfers:     TAX TAY TXA TYA TSX TXS
  Stack:         PHA PLA PHP PLP
  Arithmetic:    ADC SBC INC DEC INX DEX INY DEY
  Logic:         AND ORA EOR
  Shifts:        ASL LSR ROL ROR
  Compare:       CMP CPX CPY
  Branches:      BEQ BNE BCC BCS BPL BMI BVC BVS
  Jumps:         JMP JSR RTS
  Flag ops:      CLC SEC CLI SEI CLV CLD SED
  Misc:          NOP BRK (BRK halts, used as end-of-test marker)

Addressing modes: imm, zp, zp,x, zp,y, abs, abs,x, abs,y, (zp,x), (zp),y,
                  accumulator, implied, relative.

Usage:
    cpu = CPU(memory=bytearray(65536))
    cpu.pc = 0x0800
    cpu.run_until_brk()  # or cpu.step() / cpu.run(max_cycles=...)
"""

# Flag bits
FLAG_C = 0x01
FLAG_Z = 0x02
FLAG_I = 0x04
FLAG_D = 0x08
FLAG_B = 0x10
FLAG_U = 0x20   # always 1
FLAG_V = 0x40
FLAG_N = 0x80


class CPU:
    def __init__(self, memory=None):
        self.mem = memory if memory is not None else bytearray(65536)
        self.a = 0
        self.x = 0
        self.y = 0
        self.sp = 0xFD
        self.p = FLAG_U | FLAG_I
        self.pc = 0
        self.cycles = 0
        self.halted = False

    # ─── memory access ─────────────────────────────────────────
    def r8(self, addr):
        return self.mem[addr & 0xFFFF]

    def w8(self, addr, v):
        self.mem[addr & 0xFFFF] = v & 0xFF

    def r16(self, addr):
        return self.mem[addr & 0xFFFF] | (self.mem[(addr + 1) & 0xFFFF] << 8)

    def r16_zp(self, zp):
        """Zero-page 16-bit read (wraps within zero page)."""
        return self.mem[zp & 0xFF] | (self.mem[(zp + 1) & 0xFF] << 8)

    # ─── flag helpers ──────────────────────────────────────────
    def set_nz(self, v):
        v &= 0xFF
        self.p = (self.p & ~(FLAG_N | FLAG_Z))
        if v == 0:   self.p |= FLAG_Z
        if v & 0x80: self.p |= FLAG_N

    def get_flag(self, f): return (self.p & f) != 0
    def set_flag(self, f, v):
        if v: self.p |= f
        else: self.p &= ~f

    # ─── stack ─────────────────────────────────────────────────
    def push(self, v):
        self.mem[0x100 + self.sp] = v & 0xFF
        self.sp = (self.sp - 1) & 0xFF
    def pull(self):
        self.sp = (self.sp + 1) & 0xFF
        return self.mem[0x100 + self.sp]

    # ─── addressing helpers ────────────────────────────────────
    def _imm(self):
        v = self.r8(self.pc); self.pc = (self.pc + 1) & 0xFFFF; return v
    def _zp(self):
        a = self.r8(self.pc); self.pc = (self.pc + 1) & 0xFFFF; return a
    def _zpx(self):
        return (self._zp() + self.x) & 0xFF
    def _zpy(self):
        return (self._zp() + self.y) & 0xFF
    def _abs(self):
        lo = self.r8(self.pc); hi = self.r8((self.pc + 1) & 0xFFFF)
        self.pc = (self.pc + 2) & 0xFFFF
        return lo | (hi << 8)
    def _absx(self):
        return (self._abs() + self.x) & 0xFFFF
    def _absy(self):
        return (self._abs() + self.y) & 0xFFFF
    def _indx(self):
        zp = (self._zp() + self.x) & 0xFF
        return self.mem[zp] | (self.mem[(zp + 1) & 0xFF] << 8)
    def _indy(self):
        zp = self._zp()
        base = self.mem[zp] | (self.mem[(zp + 1) & 0xFF] << 8)
        return (base + self.y) & 0xFFFF
    def _rel(self):
        off = self._imm()
        if off & 0x80: off -= 256
        return (self.pc + off) & 0xFFFF

    # ─── ALU ops ───────────────────────────────────────────────
    def _adc(self, v):
        c = 1 if self.get_flag(FLAG_C) else 0
        a = self.a
        r = a + v + c
        self.set_flag(FLAG_C, r > 0xFF)
        # overflow: set if signs of a and v are the same and differ from result
        self.set_flag(FLAG_V, ((a ^ r) & (v ^ r) & 0x80) != 0)
        self.a = r & 0xFF
        self.set_nz(self.a)

    def _sbc(self, v):
        self._adc(v ^ 0xFF)

    def _cmp(self, reg, v):
        r = (reg - v) & 0x1FF
        self.set_flag(FLAG_C, reg >= v)
        self.set_nz(r & 0xFF)

    def _asl(self, v):
        self.set_flag(FLAG_C, (v & 0x80) != 0)
        r = (v << 1) & 0xFF
        self.set_nz(r)
        return r

    def _lsr(self, v):
        self.set_flag(FLAG_C, (v & 0x01) != 0)
        r = (v >> 1) & 0xFF
        self.set_nz(r)
        return r

    def _rol(self, v):
        new_c = (v & 0x80) != 0
        r = ((v << 1) | (1 if self.get_flag(FLAG_C) else 0)) & 0xFF
        self.set_flag(FLAG_C, new_c)
        self.set_nz(r)
        return r

    def _ror(self, v):
        new_c = (v & 0x01) != 0
        r = ((v >> 1) | (0x80 if self.get_flag(FLAG_C) else 0)) & 0xFF
        self.set_flag(FLAG_C, new_c)
        self.set_nz(r)
        return r

    # ─── one instruction ───────────────────────────────────────
    def step(self):
        if self.halted: return
        op = self.r8(self.pc)
        self.pc = (self.pc + 1) & 0xFFFF
        self.cycles += 1

        # Massive opcode dispatch. Written as a flat if/elif chain for
        # clarity — Python's bytecode cache makes this plenty fast for
        # the few million instructions the tests will run.

        # LDA
        if   op == 0xA9: self.a = self._imm();                    self.set_nz(self.a)
        elif op == 0xA5: self.a = self.r8(self._zp());            self.set_nz(self.a)
        elif op == 0xB5: self.a = self.r8(self._zpx());           self.set_nz(self.a)
        elif op == 0xAD: self.a = self.r8(self._abs());           self.set_nz(self.a)
        elif op == 0xBD: self.a = self.r8(self._absx());          self.set_nz(self.a)
        elif op == 0xB9: self.a = self.r8(self._absy());          self.set_nz(self.a)
        elif op == 0xA1: self.a = self.r8(self._indx());          self.set_nz(self.a)
        elif op == 0xB1: self.a = self.r8(self._indy());          self.set_nz(self.a)
        # LDX
        elif op == 0xA2: self.x = self._imm();                    self.set_nz(self.x)
        elif op == 0xA6: self.x = self.r8(self._zp());            self.set_nz(self.x)
        elif op == 0xB6: self.x = self.r8(self._zpy());           self.set_nz(self.x)
        elif op == 0xAE: self.x = self.r8(self._abs());           self.set_nz(self.x)
        elif op == 0xBE: self.x = self.r8(self._absy());          self.set_nz(self.x)
        # LDY
        elif op == 0xA0: self.y = self._imm();                    self.set_nz(self.y)
        elif op == 0xA4: self.y = self.r8(self._zp());            self.set_nz(self.y)
        elif op == 0xB4: self.y = self.r8(self._zpx());           self.set_nz(self.y)
        elif op == 0xAC: self.y = self.r8(self._abs());           self.set_nz(self.y)
        elif op == 0xBC: self.y = self.r8(self._absx());          self.set_nz(self.y)
        # STA
        elif op == 0x85: self.w8(self._zp(),   self.a)
        elif op == 0x95: self.w8(self._zpx(),  self.a)
        elif op == 0x8D: self.w8(self._abs(),  self.a)
        elif op == 0x9D: self.w8(self._absx(), self.a)
        elif op == 0x99: self.w8(self._absy(), self.a)
        elif op == 0x81: self.w8(self._indx(), self.a)
        elif op == 0x91: self.w8(self._indy(), self.a)
        # STX
        elif op == 0x86: self.w8(self._zp(),   self.x)
        elif op == 0x96: self.w8(self._zpy(),  self.x)
        elif op == 0x8E: self.w8(self._abs(),  self.x)
        # STY
        elif op == 0x84: self.w8(self._zp(),   self.y)
        elif op == 0x94: self.w8(self._zpx(),  self.y)
        elif op == 0x8C: self.w8(self._abs(),  self.y)
        # Transfers
        elif op == 0xAA: self.x = self.a; self.set_nz(self.x)
        elif op == 0xA8: self.y = self.a; self.set_nz(self.y)
        elif op == 0x8A: self.a = self.x; self.set_nz(self.a)
        elif op == 0x98: self.a = self.y; self.set_nz(self.a)
        elif op == 0xBA: self.x = self.sp; self.set_nz(self.x)
        elif op == 0x9A: self.sp = self.x
        # Stack
        elif op == 0x48: self.push(self.a)
        elif op == 0x68: self.a = self.pull(); self.set_nz(self.a)
        elif op == 0x08: self.push(self.p | FLAG_B | FLAG_U)
        elif op == 0x28: self.p = (self.pull() & ~FLAG_B) | FLAG_U
        # Logic
        elif op == 0x29: self.a &= self._imm();             self.set_nz(self.a)
        elif op == 0x25: self.a &= self.r8(self._zp());     self.set_nz(self.a)
        elif op == 0x35: self.a &= self.r8(self._zpx());    self.set_nz(self.a)
        elif op == 0x2D: self.a &= self.r8(self._abs());    self.set_nz(self.a)
        elif op == 0x3D: self.a &= self.r8(self._absx());   self.set_nz(self.a)
        elif op == 0x39: self.a &= self.r8(self._absy());   self.set_nz(self.a)
        elif op == 0x09: self.a |= self._imm();             self.set_nz(self.a)
        elif op == 0x05: self.a |= self.r8(self._zp());     self.set_nz(self.a)
        elif op == 0x15: self.a |= self.r8(self._zpx());    self.set_nz(self.a)
        elif op == 0x0D: self.a |= self.r8(self._abs());    self.set_nz(self.a)
        elif op == 0x1D: self.a |= self.r8(self._absx());   self.set_nz(self.a)
        elif op == 0x19: self.a |= self.r8(self._absy());   self.set_nz(self.a)
        elif op == 0x49: self.a ^= self._imm();             self.set_nz(self.a)
        elif op == 0x45: self.a ^= self.r8(self._zp());     self.set_nz(self.a)
        elif op == 0x55: self.a ^= self.r8(self._zpx());    self.set_nz(self.a)
        elif op == 0x4D: self.a ^= self.r8(self._abs());    self.set_nz(self.a)
        elif op == 0x5D: self.a ^= self.r8(self._absx());   self.set_nz(self.a)
        elif op == 0x59: self.a ^= self.r8(self._absy());   self.set_nz(self.a)
        # Arithmetic
        elif op == 0x69: self._adc(self._imm())
        elif op == 0x65: self._adc(self.r8(self._zp()))
        elif op == 0x75: self._adc(self.r8(self._zpx()))
        elif op == 0x6D: self._adc(self.r8(self._abs()))
        elif op == 0x7D: self._adc(self.r8(self._absx()))
        elif op == 0x79: self._adc(self.r8(self._absy()))
        elif op == 0x71: self._adc(self.r8(self._indy()))
        elif op == 0xE9: self._sbc(self._imm())
        elif op == 0xE5: self._sbc(self.r8(self._zp()))
        elif op == 0xF5: self._sbc(self.r8(self._zpx()))
        elif op == 0xED: self._sbc(self.r8(self._abs()))
        elif op == 0xFD: self._sbc(self.r8(self._absx()))
        elif op == 0xF9: self._sbc(self.r8(self._absy()))
        elif op == 0xF1: self._sbc(self.r8(self._indy()))
        # INC / DEC mem
        elif op == 0xE6:
            a = self._zp();  v = (self.r8(a) + 1) & 0xFF; self.w8(a, v); self.set_nz(v)
        elif op == 0xF6:
            a = self._zpx(); v = (self.r8(a) + 1) & 0xFF; self.w8(a, v); self.set_nz(v)
        elif op == 0xEE:
            a = self._abs(); v = (self.r8(a) + 1) & 0xFF; self.w8(a, v); self.set_nz(v)
        elif op == 0xFE:
            a = self._absx();v = (self.r8(a) + 1) & 0xFF; self.w8(a, v); self.set_nz(v)
        elif op == 0xC6:
            a = self._zp();  v = (self.r8(a) - 1) & 0xFF; self.w8(a, v); self.set_nz(v)
        elif op == 0xD6:
            a = self._zpx(); v = (self.r8(a) - 1) & 0xFF; self.w8(a, v); self.set_nz(v)
        elif op == 0xCE:
            a = self._abs(); v = (self.r8(a) - 1) & 0xFF; self.w8(a, v); self.set_nz(v)
        elif op == 0xDE:
            a = self._absx();v = (self.r8(a) - 1) & 0xFF; self.w8(a, v); self.set_nz(v)
        # INX/DEX/INY/DEY
        elif op == 0xE8: self.x = (self.x + 1) & 0xFF; self.set_nz(self.x)
        elif op == 0xCA: self.x = (self.x - 1) & 0xFF; self.set_nz(self.x)
        elif op == 0xC8: self.y = (self.y + 1) & 0xFF; self.set_nz(self.y)
        elif op == 0x88: self.y = (self.y - 1) & 0xFF; self.set_nz(self.y)
        # Shifts (accumulator / memory)
        elif op == 0x0A: self.a = self._asl(self.a)
        elif op == 0x06:
            a = self._zp();  self.w8(a, self._asl(self.r8(a)))
        elif op == 0x16:
            a = self._zpx(); self.w8(a, self._asl(self.r8(a)))
        elif op == 0x0E:
            a = self._abs(); self.w8(a, self._asl(self.r8(a)))
        elif op == 0x1E:
            a = self._absx();self.w8(a, self._asl(self.r8(a)))
        elif op == 0x4A: self.a = self._lsr(self.a)
        elif op == 0x46:
            a = self._zp();  self.w8(a, self._lsr(self.r8(a)))
        elif op == 0x4E:
            a = self._abs(); self.w8(a, self._lsr(self.r8(a)))
        elif op == 0x2A: self.a = self._rol(self.a)
        elif op == 0x26:
            a = self._zp();  self.w8(a, self._rol(self.r8(a)))
        elif op == 0x2E:
            a = self._abs(); self.w8(a, self._rol(self.r8(a)))
        elif op == 0x6A: self.a = self._ror(self.a)
        elif op == 0x66:
            a = self._zp();  self.w8(a, self._ror(self.r8(a)))
        elif op == 0x6E:
            a = self._abs(); self.w8(a, self._ror(self.r8(a)))
        # Compare
        elif op == 0xC9: self._cmp(self.a, self._imm())
        elif op == 0xC5: self._cmp(self.a, self.r8(self._zp()))
        elif op == 0xD5: self._cmp(self.a, self.r8(self._zpx()))
        elif op == 0xCD: self._cmp(self.a, self.r8(self._abs()))
        elif op == 0xDD: self._cmp(self.a, self.r8(self._absx()))
        elif op == 0xD9: self._cmp(self.a, self.r8(self._absy()))
        elif op == 0xD1: self._cmp(self.a, self.r8(self._indy()))
        elif op == 0xE0: self._cmp(self.x, self._imm())
        elif op == 0xE4: self._cmp(self.x, self.r8(self._zp()))
        elif op == 0xEC: self._cmp(self.x, self.r8(self._abs()))
        elif op == 0xC0: self._cmp(self.y, self._imm())
        elif op == 0xC4: self._cmp(self.y, self.r8(self._zp()))
        elif op == 0xCC: self._cmp(self.y, self.r8(self._abs()))
        # Branches
        elif op == 0x10: t = self._rel(); (setattr(self, 'pc', t) if not self.get_flag(FLAG_N) else None)
        elif op == 0x30: t = self._rel(); (setattr(self, 'pc', t) if     self.get_flag(FLAG_N) else None)
        elif op == 0x50: t = self._rel(); (setattr(self, 'pc', t) if not self.get_flag(FLAG_V) else None)
        elif op == 0x70: t = self._rel(); (setattr(self, 'pc', t) if     self.get_flag(FLAG_V) else None)
        elif op == 0x90: t = self._rel(); (setattr(self, 'pc', t) if not self.get_flag(FLAG_C) else None)
        elif op == 0xB0: t = self._rel(); (setattr(self, 'pc', t) if     self.get_flag(FLAG_C) else None)
        elif op == 0xD0: t = self._rel(); (setattr(self, 'pc', t) if not self.get_flag(FLAG_Z) else None)
        elif op == 0xF0: t = self._rel(); (setattr(self, 'pc', t) if     self.get_flag(FLAG_Z) else None)
        # Jumps
        elif op == 0x4C: self.pc = self._abs()
        elif op == 0x6C:
            a = self._abs()
            # 6502 indirect JMP bug: page-wrap on low byte, but we honor it
            lo = self.r8(a)
            hi = self.r8((a & 0xFF00) | ((a + 1) & 0xFF))
            self.pc = lo | (hi << 8)
        elif op == 0x20:
            a = self._abs()
            ret = (self.pc - 1) & 0xFFFF
            self.push((ret >> 8) & 0xFF)
            self.push(ret & 0xFF)
            self.pc = a
        elif op == 0x60:
            lo = self.pull(); hi = self.pull()
            self.pc = ((hi << 8) | lo) + 1 & 0xFFFF
        # Flag ops
        elif op == 0x18: self.set_flag(FLAG_C, False)
        elif op == 0x38: self.set_flag(FLAG_C, True)
        elif op == 0x58: self.set_flag(FLAG_I, False)
        elif op == 0x78: self.set_flag(FLAG_I, True)
        elif op == 0xB8: self.set_flag(FLAG_V, False)
        elif op == 0xD8: self.set_flag(FLAG_D, False)
        elif op == 0xF8: self.set_flag(FLAG_D, True)
        # BIT
        elif op == 0x24:
            v = self.r8(self._zp())
            self.set_flag(FLAG_Z, (self.a & v) == 0)
            self.set_flag(FLAG_N, (v & 0x80) != 0)
            self.set_flag(FLAG_V, (v & 0x40) != 0)
        elif op == 0x2C:
            v = self.r8(self._abs())
            self.set_flag(FLAG_Z, (self.a & v) == 0)
            self.set_flag(FLAG_N, (v & 0x80) != 0)
            self.set_flag(FLAG_V, (v & 0x40) != 0)
        # Misc
        elif op == 0xEA: pass
        elif op == 0x00:
            # BRK: used as end-of-test marker. Halt.
            self.halted = True
        else:
            raise NotImplementedError(f"Unknown opcode ${op:02X} at ${(self.pc-1)&0xFFFF:04X}")

    def run(self, max_cycles=50_000_000):
        while not self.halted and self.cycles < max_cycles:
            self.step()
        if self.cycles >= max_cycles:
            raise RuntimeError(f"cycle limit hit at pc=${self.pc:04X}")


# ─── Self-test ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Quick sanity: add two bytes, store result, BRK.
    cpu = CPU()
    prog = bytes([
        0xA9, 0x05,         # LDA #$05
        0x69, 0x03,         # ADC #$03     (carry clear → 8)
        0x85, 0x10,         # STA $10
        0x00,               # BRK
    ])
    cpu.mem[0x0800:0x0800 + len(prog)] = prog
    cpu.pc = 0x0800
    cpu.run()
    assert cpu.mem[0x10] == 8, f"expected 8, got {cpu.mem[0x10]}"
    print(f"ADC test: mem[$10] = {cpu.mem[0x10]}  OK")

    # Signed compare via SBC
    cpu = CPU()
    # 100 - 50 = 50, C=1 (no borrow), N=0
    prog = bytes([
        0xA9, 0x64,         # LDA #100
        0x38,               # SEC
        0xE9, 0x32,         # SBC #50
        0x85, 0x20,         # STA $20
        0x00,
    ])
    cpu.mem[0x0800:0x0800 + len(prog)] = prog
    cpu.pc = 0x0800
    cpu.run()
    assert cpu.mem[0x20] == 50
    assert cpu.get_flag(FLAG_C)
    print(f"SBC test: mem[$20] = {cpu.mem[0x20]}, C={cpu.get_flag(FLAG_C)}  OK")

    # JSR/RTS round trip
    cpu = CPU()
    prog = bytes([
        0xA9, 0x01,         # $0800: LDA #1
        0x20, 0x10, 0x08,   # $0802: JSR $0810
        0x85, 0x30,         # $0805: STA $30
        0x00,               # $0807: BRK
    ])
    cpu.mem[0x0800:0x0800 + len(prog)] = prog
    # Subroutine at $0810: A = A + 2, RTS
    cpu.mem[0x0810:0x0814] = bytes([0x69, 0x02, 0x60, 0x00])
    # Need carry clear before ADC
    cpu.p = FLAG_U
    cpu.pc = 0x0800
    cpu.run()
    assert cpu.mem[0x30] == 3, f"expected 3, got {cpu.mem[0x30]}"
    print(f"JSR test: mem[$30] = {cpu.mem[0x30]}  OK")

    # (zp),Y load/store
    cpu = CPU()
    cpu.mem[0x0050:0x0052] = bytes([0x00, 0x30])   # pointer = $3000
    cpu.mem[0x3005] = 0x42
    prog = bytes([
        0xA0, 0x05,         # LDY #5
        0xB1, 0x50,         # LDA ($50),Y
        0x85, 0x40,         # STA $40
        0x00,
    ])
    cpu.mem[0x0800:0x0800+len(prog)] = prog
    cpu.pc = 0x0800
    cpu.run()
    assert cpu.mem[0x40] == 0x42
    print(f"(zp),Y test: mem[$40] = ${cpu.mem[0x40]:02X}  OK")

    print("\nAll 6502 interpreter self-tests passed.")
