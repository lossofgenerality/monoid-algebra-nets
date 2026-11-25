# monoid-algebra-nets

*Updated 24 November 2026 · Computational Classnotes*

These are neural nets that **parse, evaluate and simplify** tensor-algebra expressions written with ⨂ (tensor product) and ⨁ (direct sum), for example `(a⨁b)⨂c`. Everything is in Wolfram Language, as notebooks with one Input cell each. The move to Python comes later.

By Dara O Shayda and Hikari Noor Hoshino (星野ヌール)  
AI System Programming Laboratory

## The design

There are three separate nets, each paired with an exact symbolic counterpart that labels its training data and checks its answers:

| Net | Task | Kind |
|---|---|---|
| `tnParserNet` | parse | transformer encoder–decoder, tokens → prefix tree |
| `tnPolicyNet` | simplify | transformer encoder with two heads: *which rule*, *at which node* |
| `tnValueNet` | evaluate | transformer encoder, pooled regression |

The nets decide and the exact engine executes, so every rewrite is sound. The full design is in **`DESIGN-tensornets.md`** (and `.pdf`). It includes Appendix A, which explains each term with examples, and its references.

## Notebooks

| Notebook | Holds |
|---|---|
| `ff_tensornets_new.nb` | **The current code**: every function in one cell. Evaluating it writes nothing to the Wolfram Cloud. |
| `ff_tensornets_test.nb` | The tests: 22 of 22 pass |
| `ff_tensornets_new_testlog.nb` | The latest test results |
| `ff_tensornets_smallnets.nb` | Small development nets (wiring checks only); evaluate after `ff_tensornets_new.nb` |
| `ff_tensornets.nb`, `ff_tensornets_testlog.nb` | The earlier version and its test log, kept for the record |
| `ref1_seq2seq_sutskever2014.nb` | Reference [1]: Sutskever, Vinyals and Le, *Sequence to Sequence Learning with Neural Networks* (2014). The paper's deep-LSTM model, with a small demo |
| `ref2_symbolic_math_lample_charton2019.nb` | Reference [2]: Lample and Charton, *Deep Learning for Symbolic Mathematics* (2019). The paper's method (BWD data, prefix notation, verified beam search), with a small demo that learns integration |
| `ref3_transformer_vaswani2017.nb` | Reference [3]: Vaswani et al., *Attention Is All You Need* (2017). The base Transformer, with a small demo |
| `ref*_log.nb` | The demo results of each reference notebook |
| `ref*_nets.nb` | **The trained demo nets** of each reference notebook: one cell, each net stored as `name = Uncompress["…"];` |

**Demo results** (small nets, CPU, Wolfram 15.0, 23 November 2026; each in its own fresh process):

| Notebook | Demo | Result |
|---|---|---|
| ref1 (deep LSTM seq2seq) | infix ⨁/⨂ → prefix form, 2 × 128 LSTM, 30 rounds | exact match 80.5 % (beam 3) with the source reversed, 76.5 % without: the paper's finding, on a small scale |
| ref2 (Lample and Charton) | integration learnt from BWD pairs (f′ → f), Transformer 2 layers × 128 | 86.5 % (beam 1), 95.5 % (beam 5) of 200 held-out functions, each answer **verified by differentiation** |
| ref3 (Transformer base) | infix ⨁/⨂ → prefix form, N = 2, d = 64, 20 rounds | exact match 84.5 % (greedy), 85.0 % (beam 4, α = 0.6) |

To use a saved demo net without training again (ref3 shown; ref1 and ref2 use `s2sSkipDemo` and `lcSkipDemo`):
```
tfSkipDemo = True
(* evaluate ref3_transformer_vaswani2017.nb, then ref3_transformer_vaswani2017_nets.nb *)
tfBeam[tfNet, {"(", "a", "⨁", "b", ")", "⨂", "c"}, 4]      (* {"P", "S", "a", "b", "c"} *)
```
The nets saved are `s2sNetReversed` and `s2sNetForward` (ref1), `lcNet` (ref2) and `tfNet` (ref3). Each was checked by restoring it in a fresh process.

The paper-sized models are defined and their sizes counted, but not trained: 384.1M parameters for ref1 (the paper reports 384M) and 63.3M for ref3 (the paper reports 65M).

The papers themselves are not included; they are freely available on arXiv (1409.3215, 1912.01412, 1706.03762).

## Units of learning

Four units explain how the nets are made, in the format of the Laplacian write-ups. Each unit has a **write-up** (HTML, PDF, Markdown) and a **notebook** with the same section numbers. The notebook starts with one toolkit cell, then one section per idea. Each section has an explanation, a worked example done by hand and a check that the kernel returns. The notebooks are saved with their outputs, so they can be read without running them.

| Unit | Write-up | Notebook | Follows |
|---|---|---|---|
| 1 · Reading a Sequence, Writing a Sequence | [HTML](docs/units/01-seq2seq.html) · [PDF](docs/units/01-seq2seq.pdf) · [md](docs/units/01-seq2seq.md) | `unit1_seq2seq.nb` | Sutskever, Vinyals and Le (2014) |
| 2 · Mathematics as Data | [HTML](docs/units/02-symbolic-data.html) · [PDF](docs/units/02-symbolic-data.pdf) · [md](docs/units/02-symbolic-data.md) | `unit2_symbolic_data.nb` | Lample and Charton (2019) |
| 3 · Attention, Part by Part | [HTML](docs/units/03-transformer.html) · [PDF](docs/units/03-transformer.pdf) · [md](docs/units/03-transformer.md) | `unit3_transformer.nb` | Vaswani et al. (2017) |
| 4 · Building Our Own Net | [HTML](docs/units/04-build-your-own.html) · [PDF](docs/units/04-build-your-own.pdf) · [md](docs/units/04-build-your-own.md) | `unit4_build_your_own.nb`, trained net in `unit4_build_your_own_nets.nb` | Units 1–3, on ⨂/⨁ expressions |

Every technical term is defined in the [glossary](docs/glossary.html); the [units index](docs/units/index.html) links everything. The write-ups are built by `tools/units/build_docs.py` and the glossary and index by `tools/units/build_glossary.py`.

To use a unit, open its notebook in Wolfram, choose Evaluation ▸ Quit Kernel ▸ Local, and evaluate it from top to bottom. Unit 4 trains its net in about five minutes; to skip that, evaluate §1 and then `unit4_build_your_own_nets.nb`.

## Lessons (superseded by the units)

The earlier single-cell lesson notebooks, kept for the record until their removal is approved. They code the important parts of references [1]–[3] from plain arrays, check each part against Wolfram's own layers, and then put them together on our own problem. Each has one Input cell; evaluating it writes a log notebook with the explanation, the formulas, the results and a PASS/FAIL line for every check. Nothing is printed and nothing is written to the Wolfram Cloud.

| Notebook | Covers | Checks |
|---|---|---|
| `lesson1_lstm_seq2seq.nb` | [1] The LSTM cell, the deep encoder and its thought vector, the decoder step, why reversing the source helps, uniform initialisation, gradient-norm clipping, the learning-rate schedule, beam search, the 384M parameter count | 19 of 19 pass |
| `lesson2_symbolic_data.nb` | [2] Prefix notation and integers, counting trees and expressions, uniform tree sampling (Algorithms 1 and 2) against a naive sampler, FWD/BWD/IBP data, verified hypotheses | 21 of 21 pass |
| `lesson3_transformer_parts.nb` | [3] Scaled dot-product attention, why √dk, masks, multi-head attention built from primitive layers, sinusoidal positions as a net, feed-forward, LayerNorm(x + Sublayer(x)), encoder and decoder layers, shared embeddings, label smoothing, warm-up, length penalty, the parameter count | 27 of 27 pass |
| `lesson4_build_your_own.nb` | Our own: a transformer assembled from Lesson 3's parts that expands ⨂/⨁ expressions, trained with label smoothing and warm-up, decoded with beam search and the prefix grammar, every answer checked by an exact oracle | 9 of 9 pass |
| `lesson*_log.nb` | The results of each lesson | |
| `lesson4_build_your_own_nets.nb` | The trained net of Lesson 4, as `owNet = Uncompress["…"];` | restored and checked in a fresh process |

**Lesson 4 results** (d = 32, 4 heads, 2 + 2 layers, 12,000 training pairs, 40 rounds, 211 s on the CPU; 100 held-out sources):

| Decoding | Valid prefix | Top answer verified | Any in beam verified |
|---|---|---|---|
| greedy | 99 % | 93 % | 93 % |
| beam 4, α = 0.6 | 100 % | 94 % | 99 % |
| beam 4 + prefix grammar | 100 % | 94 % | 99 % |

**A finding from Lesson 2.** Equation 2 of Lample and Charton, as printed, ends in −p₁(n−2)E₍ₙ₋₂₎. Expanding (1 − p₁z)² in their generating function (Appendix B) gives p₁² there. The printed form is right only when p₁ is 0 or 1, the cases the paper checks; Lesson 2 shows the difference against a count of every tree.

To use the trained net of Lesson 4:
```
owSkipRun = True
(* evaluate lesson4_build_your_own.nb, then lesson4_build_your_own_nets.nb *)
First@First@owBeam[owNet, {"(", "a", "⨁", "b", ")", "⨂", "c"}, 4, True]   (* {"⨁", "⨂", "a", "c", "⨂", "b", "c"} *)
```

## Related

- `python/tensor_parser.py` and `python/CONTEXT.md`: a pure-Python parser for the same notation.
- **Net diagrams** ([`docs/net-diagrams.html`](https://lossofgenerality.github.io/monoid-algebra-nets/docs/net-diagrams.html)): the CCN Net Diagrams editor, for drawing a small neural net and generating a Wolfram notebook or a Python script that trains it. The editor lives in the CCN Net Lab and is published in `lossofgenerality/ccn-nets` (https://lossofgenerality.github.io/ccn-nets/editor/net-diagram-editor.html). This site shows the live copy, so a change there appears here at once, with no copy to keep in step. Under Examples it opens owNet (Unit 4) and the three TensorNets nets, drawn from their definitions; it counts their weights as Wolfram does (owNet 59,950, tnParserNet 1,422,870, tnPolicyNet 611,468, tnValueNet 626,819) and writes each as a Wolfram `NetGraph`. The page is made by `tools/units/build_glossary.py`.

## Use

1. Open `ff_tensornets_new.nb` and evaluate its cell.
2. Try, for example:
   ```
   tnSimplify["(a⨁b)⨂c", "Expand"]
   tnEvaluate["(V⨁W)^2", <|"V" -> 2, "W" -> 3|>]
   ```
3. To train the full-size nets, run `tnTrainAll[]`; progress goes to a log notebook.

Nothing is written to the Wolfram Cloud unless you call `tensorNetsSave[]` (the release) or `tnSaveNets[]` (the nets).

---

© Computational Classnotes · CC BY-NC-SA 4.0
