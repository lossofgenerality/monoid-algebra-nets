# tensor-algebra-nets

*Updated 23 November 2026 · Computational Classnotes*

These are neural nets that **parse, evaluate and simplify** tensor-algebra expressions written with ⨂ (tensor product) and ⨁ (direct sum), for example `(a⨁b)⨂c`. Everything is in Wolfram Language, as notebooks with one Input cell each. The move to Python comes later.

By Dara O Shayda and Hikari Noor Hoshino (星野ヌール), AI System Programmer

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

## Related

- `python/tensor_parser.py` and `python/CONTEXT.md`: a pure-Python parser for the same notation.

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
