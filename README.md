# Soul Player C64

**A real transformer running on a 1 MHz Commodore 64.**

```
  .--------.
  | O    O  |
  |    V    |
  |..|---|..|

SOUL PLAYER C64
2026-04-06 - GIZMO64K

25k PARAMETERS
REAL TRANSFORMER. REAL WEIGHTS.
LOADED OFF A FLOPPY DISK.

2 LAYERS, 4 HEADS, REAL ATTENTION,
INT8, 64K!

TYPE AND I WILL TALK BACK
EVENTUALLY... AFTER MINUTES!

YOU> hey
C64> HELLO! RE SOUNDS ME. MEFUL!
YOU> omg it works, this makes me so happy
C64> I ME
YOU> you what? :D
C64> I HAVE HAVE ME.
YOU> 
```

A 2-layer decoder-only transformer... the same architecture behind ChatGPT, Claude, and Gemini... implemented in hand-written 6502 assembly and running on an unmodified Commodore 64. ~25,000 int8 parameters. Real multi-head attention, real softmax, real RMSNorm. Roughly one token per minute on the actual 1 MHz chip. The whole program fits on a floppy disk with room to spare.

## Quick start

Grab `disk/soulplayer.d64` and run it in any C64 emulator (VICE recommended) or flash it to a 1541 disk image for real hardware:

```
LOAD"SOULPLAYER",8,1
RUN
```

Wait for the banner, type a short message, press RETURN, **wait**. The border will flash through colors while bob_soul_v3 thinks. Each generated token is announced with a short SID blip. A 6-8 token response takes several minutes. Bring a book.

Type `q` and RETURN to quit.

**What to type:** bob_soul_v3 only understands lowercase ASCII letters, spaces, and basic punctuation (`.`, `,`, `!`, `?`). Capital letters and anything else become `<UNK>` tokens, which confuse it.

## What's in this repo

```
soul-player-c64/
├── README.md                 — you are here
├── disk/
│   ├── soulplayer.d64        — ready-to-run disk image
│   └── soulplayer.prg        — raw PRG (load at $0801)
├── build/
│   ├── bob_soul_v3.bin       — trained model weights (25,760 bytes)
│   └── bob_tokenizer_v2.json — BPE tokenizer (128 tokens)
├── src/
│   ├── soon!
```



## Architecture

```
Vocab size:      128 tokens (4 specials + 30 chars + 90 BPE merges)
Embedding dim:   32
Layers:          2
Attention heads: 4 heads × 8 dims per head
FFN hidden:      64
Max sequence:    20 tokens
Parameters:      ~25,000 (all int8 with per-tensor shifts)
```

Each layer is: pre-norm RMSNorm -> multi-head causal self-attention -> residual -> RMSNorm -> 2-layer MLP with ReLU -> residual. Final RMSNorm, output projection, argmax. No sampling -> friendly, but greedy decoding only.

All activations are Q8.8 fixed-point (int16). All weights are int8 with a per-tensor shift baked in at quantization time. Biases are pre-scaled to int16 in the matmul accumulator's scale. Softmax uses a 128-entry exp lookup table. RMSNorm uses integer sqrt and integer division. Division is restoring; multiplication is shift-and-add. And, who would have thought?.. The 6510 has no multiply instruction and no floating point unit... so every arithmetic operation is built out of loads, stores, shifts, adds, and branches.. normal! ;)

memory layout on the C64:

```
$0801-$2088   code + tokenizer tables           (~6 KB)
$2100-$85A0   weights (loaded from PRG)         (25.3 KB)
$8600-$9F00   activation buffers                (6.5 KB)
$A000-$BFFF   BASIC ROM (not touched)
$C000-$C3FF   token buffer, input, logits, scratch
$D000-        VIC-II, SID, CIA (I/O)
$E000-$FFFF   KERNAL ROM
```

## Training your own Soul

(coming soon!)

## Caveats

- **bob_soul_v3 is not smart.** It has ~25,000 parameters. That's roughly 70 million times smaller than GPT-4. It will produce broken sentences and strange word combinations. This is by design — the point is that the architecture works at all at this scale, not that it produces good output.
- **Inference is slow.** On a real C64 at 1 MHz, one forward pass takes at least 60-90 seconds. A full response is several minutes.
- **Capital letters become `<UNK>`.** The vocabulary is all lowercase. Typing "HEY" gives you three unknown tokens. Stick to lowercase.
- **The model is trained on a tiny dataset.** It will tend to repeat words and phrases from its training corpus. Feel free to train your own.

## Credits

- **Code, architecture, training, slow and fast thinking... and swearing:** [gizmo64k](http://www.indiepixel.de/)
- **99% of the 6510 Code, debugging partner, rubber ducky, tremendous help, 100% of all unit tests:** Claude (Opus 4.6) by Anthropic
- **Lucky soul, now AI powered:** The Commodore 64 by Commodore Business Machines, 1982

## License

GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007. If you make something cool, tell me.

---

*The future came back for the past. And now it has a soul.*

*You can run your own AI chatbot on your own hardware! No excuses!*

*When I am lonely I talk to my Commodore!*

*64k should be enough for every bot*

*C64 - thinks for itself*

*C64 - the most intelligent homecomputer of the 1980s!*

*Who needs NVIDIA anyway?*

*Quit your Chatbot subscription - use your Commodore instead!*
