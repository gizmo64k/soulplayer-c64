# Soul Player C64

**A real transformer running on a 1 MHz Commodore 64.**
**And apparently on the Amstrad CPC, too! -> https://github.com/G1D30N/soulplayer-cpc **
```
   .-------.
  | O     O |
  |    V    |
  |..|---|..|

# SOUL PLAYER C64

25K PARAMETERS. 2 LAYERS. REAL TRANSFORMER.
LOADED OFF A FLOPPY DISK.

YOU> hey
C64> HELLO! RE SOUNDS ME. MEFUL!
```

A 2-layer decoder-only transformer - the same architecture behind ChatGPT, Claude, and Gemini - implemented in hand-written 6502/6510 assembly and running on an unmodified Commodore 64. ~25,000 int8 parameters. Real multi-head causal self-attention, real softmax, real RMSNorm. About 60 seconds per token. The whole thing fits on a floppy disk with room to spare.

## Architecture

2 layers, 4 attention heads × 8 dims, 32-dimensional embeddings, 64 FFN hidden units. ~25,000 parameters quantized to int8 with per-tensor shift scaling. The key breakthrough was fixing the softmax score normalization - shifting attention scores by 14 bits instead of 17 gives the 128-entry exp lookup table enough dynamic range to produce meaningful attention weights. Without this fix, the integer attention was essentially uniform across all positions, making the model blind regardless of architecture or training.

## Quick start - run the pre-built soul

Grab `disk/soulplayer.d64` and load it in any C64 emulator ([VICE](https://vice-emu.sourceforge.io/) recommended):

```
LOAD"SOULPLAYER",8,1
RUN
```

Type a short message in lowercase, press RETURN, wait. The border flashes while it thinks. Each token gets a SID blip. A full response takes a few minutes. Type `q` to quit.

> **Tip:** The model understands lowercase letters, spaces, and punctuation (`. , ! ? ' : ; -`). Capital letters become unknown tokens.

## Train your own soul

This is the fun part. Write a corpus, train a model, build a floppy.

### Install dependencies

```bash
pip install numpy torch
```

### Write a corpus

Create a text file with one exchange per line in `<SEP>input<SEP>response<SEP>` format:

```
<SEP>hello<SEP>hey! nice to see you!<SEP>
<SEP>i'm sad<SEP>i hear you. i care about you.<SEP>
<SEP>tell me a joke<SEP>why did the bit flip? it was tired!<SEP>
```

Keep exchanges short - the model has a 20-token context window. See `data/example_corpus.txt` for a starter.

### Train

```bash
python train.py data/example_corpus.txt
```

This trains a BPE tokenizer (128 tokens), trains the QAT transformer, exports `models/soul.bin` and `models/tokenizer.json`. Takes a few minutes on GPU.

Every 500 epochs, you'll see both the **float** and **int8** inference output side by side - what the model learned vs what the C64 will actually produce. The best checkpoint is saved based on int8 quality, not float loss. All checkpoints are saved to `models/checkpoints/` for cherry-picking.

Options:
```bash
python train.py data/my_corpus.txt --epochs 30000 --output models/
python train.py                    # uses built-in emotional support corpus
```

Training resumes automatically if checkpoints exist from a previous run.

### Build the C64 binary

```bash
python build.py
```

This assembles all 6502/6510 routines, embeds your trained weights, and writes `disk/soulplayer.prg` and `disk/soulplayer.d64`.

### Run it

```bash
x64 disk/soulplayer.d64    # VICE emulator
```

Or flash the `.d64` to a real 1541 floppy for hardware.

## Chat with the soul locally

```bash
python soulchat.py                   # uses models/soul.bin
python soulchat.py models/soul.bin   # custom soul
```

Runs the same integer arithmetic as the C64, just faster.

## Run the tests

```bash
python test.py           # full suite (~90 tests, ~30 seconds)
python test.py --quick   # skip 6502/6510 assembly tests
```

Tests verify the entire chain: float reference → integer reference → memory-faithful shadow → 6502/6510 assembly routines → build round-trip.

## What's in the repo

```
soulplayer-c64/
├── train.py              - train a model + export weights
├── build.py              - assemble the C64 binary
├── test.py               - run all tests
├── soulchat.py           - chat in your terminal
│
├── data/
│   └── example_corpus.txt
├── models/
│   ├── soul.bin           - pre-trained weights (25KB, int8)
│   ├── tokenizer.json     - BPE tokenizer (128 tokens)
│   └── checkpoints/       - all saved training checkpoints
├── disk/
│   ├── meful.d64          - original release, disk image
│   └── meful.prg          - original release, raw PRG
│   ├── soulplayer.d64     - ready-to-run disk image
│   └── soulplayer.prg     - raw PRG
└── src/                   - the engine
    ├── numerics.py        - ground truth: fixed-point math + forward pass
    ├── soul_io.py         - .bin weight file format
    ├── shadow.py          - memory-faithful Python shadow of the 6502/6510
    ├── assembler.py       - mini 6502 assembler (labels, patches, far branches)
    ├── cpu6502.py         - minimal 6502 interpreter for testing
    ├── asm_matvec.py      - 6502 matrix-vector multiply
    ├── asm_rms_norm.py    - 6502 RMSNorm (integer sqrt + divide)
    ├── asm_attn_head.py   - 6502 attention head (LUT softmax)
    ├── asm_simple.py      - 6502 embed, residual, relu, argmax
    └── build.py           - PRG + D64 assembler
```

## Specs

| | |
|---|---|
| Vocab | 128 tokens (4 special + 34 chars/punct + 90 BPE merges) |
| Embedding | 32 dimensions |
| Layers | 2 |
| Attention | 4 heads × 8 dims per head |
| FFN | 64 hidden units |
| Context | 20 tokens |
| Parameters | ~25,000 (all int8) |
| Weight size | 25 KB |
| Decoding | Greedy (argmax) |

Each layer: RMSNorm → multi-head causal self-attention → residual → RMSNorm → ReLU MLP → residual. Final RMSNorm → output projection → argmax.

All activations are Q8.8 fixed-point (int16). Weights are int8 with per-tensor power-of-2 shifts. Biases are int16 pre-scaled to the matmul accumulator. Softmax uses a 128-entry exp lookup table with >>14 score normalization. The 6502 has no multiply instruction - everything is shift-and-add.

### Memory map

```
$0801-$20FF   code + tokenizer tables        (~6 KB)
$2100-$85A0   weights                       (25.3 KB)
$8600-$9D00   activation buffers             (5.8 KB)
$C000-$C3FF   token buffer, input, scratch
$D000-        VIC-II, SID, CIA (I/O)
```

### How training works

The model uses quantization-aware training (QAT). During training, weights pass through `FakeQuantI8` - fake-quantized with continuous float scaling and straight-through gradient estimation. The deliberate mismatch between training's continuous scale and export's power-of-2 shift grid acts as implicit noise, forcing the model to learn weights with wider logit margins that survive the quantization gap. Biases are fake-quantized with simple `fq()`. Every matmul gets a `× 0.5` post-shift simulating the 6502's `>> 1`.

Label smoothing (0.15) prevents the model from sharpening logit distributions beyond what int8 arithmetic can reliably distinguish. The training loop evaluates the actual integer forward pass (`numerics.forward()`) every 500 epochs and saves the best checkpoint by int8 argmax accuracy, not float loss.

The training output shows float and int8 inference side by side - what the model learned vs what the C64 will produce.

## Caveats

- **It's not smart.** 25K parameters is about 70 million times smaller than GPT-4. It will produce broken sentences. That's the point - the architecture works at this scale.
- **It's ~slow~ contemplative.** About 60 seconds per token on real hardware. A full response takes several minutes.
- **Capitals become `<UNK>`.** Stick to lowercase.
- **Small vocabulary.** 128 tokens and 20-token context - keep training exchanges short.

## Credits

- **Code, training:** [gizmo64k](http://www.indiepixel.de/)
- **Debugging, unit tests, rubber duck:** Claude (Opus 4.6) by Anthropic
- **Lucky soul:** The Commodore 64 by Commodore Business Machines, 1982

## License

GNU General Public License v3. See [LICENSE](LICENSE).

---

*The future came back for the past. And now it has a soul.*
