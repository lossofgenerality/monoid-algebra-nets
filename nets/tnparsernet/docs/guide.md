---
title: "tnParserNet: User Guide"
subtitle: "The TensorNets parser: a transformer that reads ⨂/⨁ expressions in any spelling and writes their prefix tree, checked by an exact parser"
author: "Computational Classnotes"
date: "25 November 2026"
---

# tnParserNet: User Guide

*The TensorNets parser: a transformer that reads ⨂/⨁ expressions in any spelling and writes their prefix tree, checked by an exact parser*

Computational Classnotes · 25 November 2026

**Companion to:** TensorNets (`DESIGN-tensornets.md`), the TensorNets units of learning and the CCN Net Lab catalogue (https://lossofgenerality.github.io/ccn-nets/tensor-algebras/tnparsernet/)

**Files covered:** `ff_tensornets_new.nb`, `ff_tensornets_test.nb`, `nets/tnparsernet/train/tnParserNetFromDiagram.nb`, `python/tensor_parser.py`

© Computational Classnotes · CC BY-NC-SA 4.0

## Contents

1. What it is
2. Before you start
3. Finding your way around
4. The mathematics
5. The configuration, part by part
6. Using it in Wolfram
7. Python
8. Training it yourself
9. Results
10. A complete worked session
11. Messages you may see
12. Quick reference card
13. Files
14. Credits, copyright and licence

---

## 1. What it is

tnParserNet reads an expression such as `(a ⊕ b)(x) c`, in any of its spellings, and writes its tree in prefix (Polish) form, `P S x1 x2 x3`. It is a transformer encoder–decoder with d = 128, 4 heads, a feed-forward width of 512 (GELU) and 3 encoder and 3 decoder layers: **1,422,870 numbers**, all learnt.

Its oracle is `tnRefParse`, a recursive-descent parser. `tnParse` runs the net and checks it against the oracle: with `Method -> Automatic` the exact tree is returned whenever the two differ, and the disagreement is reported.

### TensorNets in one paragraph

TensorNets splits work on ⨂/⨁ expressions into three nets, since the three tasks differ in kind: parsing is sequence-to-sequence translation (`tnParserNet`), simplification is a sequence of decisions (`tnPolicyNet`) and evaluation is regression (`tnValueNet`). Each net is paired with an exact symbolic counterpart, its *oracle*, which labels its training data and checks its answers. The nets decide; the exact engine executes.

| Task | Net | Oracle |
|---|---|---|
| Parse | `tnParserNet`: encoder–decoder, tokens → prefix tree | `tnRefParse` (recursive descent, as in `tensor_parser.py`) |
| Simplify | `tnPolicyNet`: encoder, two heads, *which rule* and *at which node* | `tnOracleStep` (the first applicable rule, in a preorder scan) |
| Evaluate | `tnValueNet`: encoder, pooled regression | `tnDimension`, `tnTermCount`, `tnNumeric` |

## 2. Before you start

### What you need

| For | You need |
|---|---|
| Wolfram | Wolfram 14 or later (tested on Wolfram 15.0). Nothing else. |
| Python | Python 3.9 or later, for the exact counterpart `python/tensor_parser.py`. NumPy only for its `evaluate`. |
| Training at full size | about an hour or more on a CPU for all three nets |

### Opening it

1. Open `ff_tensornets_new.nb`, choose Evaluation ▸ Quit Kernel ▸ Local, and evaluate its one cell. Every function is defined; nothing is printed and **nothing is written to the Wolfram Cloud**.
2. Without trained nets, every function still works: it uses the exact engine (the oracles) and says so (`"Method" -> "Oracle"` or `"Reference"`).
3. To train the three nets, run `tnTrainAll[]`. Progress goes to `ff_tensornets_log.nb`, beside the notebook.
4. To keep the trained nets, run `tnSaveNets[]`, and later `tnLoadNets[]`. These, and the release `tensorNetsSave[]`, are the only calls that write to the Cloud, and only when you make them.

**Status, 25 November 2026.** The design is drawn and tested (22 of 22 checks pass), and Wolfram's count of the net is confirmed. **Full-size training has not yet been done.**

## 3. Finding your way around

| You want to | Go to |
|---|---|
| Use TensorNets | §6, with the examples as the kernel returns them |
| Understand what tnParserNet computes | §4 and §5 |
| Work in Python | §7 |
| Train it | §8 |
| Learn how the parts are made | Units 1–4 in `docs/units/`, and Appendix A of `DESIGN-tensornets.md` |
| See or change the design | the CCN Net Diagram Editor, Examples ▸ tnParserNet |

## 4. The mathematics

### Attention and the blocks

Each head computes

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^{\mathsf T}}{\sqrt{d_k}}\right)V,\qquad d_k=d/h=32,
$$

and the four heads are joined and projected back to d = 128. Every sub-layer is wrapped as LayerNorm(x + Sublayer(x)). The feed-forward layer is W₂ GELU(W₁x + b₁) + b₂, applied at every position. Tokens are embedded in 128 dimensions and a learnt position embedding (up to 96 positions) is added.

### The lexer and the atom slots

`tnLex` is deterministic. Identifiers become atom slots `x1 … x8` in order of first appearance, so the net learns structure rather than names: `(p⨁q)⨂p` and `(a⨁b)⨂a` are the same input. Operator spellings are kept as distinct tokens (`⨁ ⊕ (+) +`, `⨂ ⊗ (x) *`, `^ ^⊗ ^⨂`), because recognising them is part of the net's task. `0`/`𝟎` is the zero space and `1`/`𝟙` the unit space. Powers become tokens such as `n2`.

### The tree and its prefix form

The tree has binary heads `tnS[a, b]` (⨁), `tnP[a, b]` (⨂) and `tnE[n, a]` (tensor power), and the leaves `tnX[i]`, `tnZero` and `tnUnit`. Binary heads make the prefix form unambiguous: (a⨁b)⨂c → `P S x1 x2 x3`. Associativity is a rewrite rule, not part of parsing. Token k of the prefix form is node k of the tree in preorder.

### Teacher forcing

The decoder is trained to predict each prefix token from the source and the tokens before it:

$$
L=-\sum_{t}\log p\left(y_t \mid y_{<t},\ \mathbf x\right),
$$

on random trees written in random spellings (random glyphs, spacing and redundant parentheses).

### Grammar-constrained decoding

Decoding is greedy, with a counter of subtrees still owed: it starts at 1, a binary head adds 1 and a leaf takes 1 away. `</s>` is forbidden while the counter is above 0, and atom slots beyond those in the input are forbidden. The output is therefore always a well-formed tree.

## 5. The configuration, part by part

### The shared parts (d = 128, 4 heads, feed-forward 512, GELU)

| Part | Formula | Numbers |
|---|---|---|
| attention layer | 4 × (128² + 128): query, key, value and output projections | 66,048 |
| feed-forward layer | 128·512 + 512 + 512·128 + 128 | 131,712 |
| LayerNorm | 2 × 128 (scale and shift) | 256 |
| encoder block | attention + feed-forward + 2 LayerNorms | 198,272 |
| encoder, 3 blocks | | 594,816 |
| decoder block | 2 attentions (causal self, cross) + feed-forward + 3 LayerNorms | 264,576 |
| learnt positions | 96 positions × 128 | 12,288 |

### tnParserNet

| Part | Numbers |
|---|---|
| input embedding: 32 tokens × 128, plus positions 96 × 128 | 16,384 |
| target embedding: 22 tokens × 128, plus positions 96 × 128 | 15,104 |
| encoder, 3 blocks | 594,816 |
| decoder, 3 blocks | 793,728 |
| output: 22 × 128 + 22, softmax per position | 2,838 |
| **Total** | **1,422,870** |

The input vocabulary (32) holds the lexer's tokens: brackets, every operator spelling, the atom slots, 0 and 1, and the power tokens. The output vocabulary (22) holds the prefix tokens with `<s>` and `</s>`.

| Setting (the defaults) | Value |
|---|---|
| d, heads, feed-forward, layers | 128, 4, 512 GELU, 3 + 3 |
| positions | learnt, up to 96 |
| ports | `Input` (source tokens), `Target` (prefix tokens with `<s>` first and `</s>` last), `Probabilities`, `Loss` |

Wolfram's `Information[NetInitialize[net], "ArraysTotalElementCount"]` returns **1,422,870** for the net in `nets/tnparsernet/train/tnParserNetFromDiagram.nb` (checked 25 November 2026).

## 6. Using it in Wolfram

### Running

```wolfram
tnParse["(a ⊕ b)(x) c"]                  (* the net if trained, checked by the oracle *)
tnParse["(a⨁b)⨂c", Method -> "Net"]       (* the net's own answer *)
tnLex["(x1 + y)*z^2"]                     (* what the net reads *)
```

### Reading the output

| Key | Holds |
|---|---|
| `"Tree"` | the tree in `tnS`, `tnP`, `tnE`, `tnX` |
| `"Atoms"` | the names behind `x1, x2, …`, in order of first appearance |
| `"String"` | the canonical string, with ⨁ and ⨂ |
| `"Prefix"` | the prefix tokens |
| `"Method"` | `"Net"` or `"Reference"` |
| `"NetAgreesWithReference"` | True, False, or `Missing["NoNet"]` |

**Try this.** `tnParse["a ⊗ (b (+) c)"]` gives `"a ⨂ (b ⨁ c)"` and the prefix `{"P", "x1", "S", "x2", "x3"}`: two different spellings, one tree.

### The public functions

| Function | Does |
|---|---|
| `tnParse[s]`, `tnParse[s, Method -> "Net"]` | string → tree, atoms, canonical string, prefix and the method used |
| `tnSimplify[s, goal]` | goal `"Expand"`, `"Factor"` or `"Simplify"`: the result, every step (rule, node, result) and a verification flag |
| `tnEvaluate[s, dims]` | the exact dimension and number of terms, and the value net's estimate; with matrices, the numeric value |
| `tnSolve[s, goal, dims]` | all of the above, JSON-ready; `dims` is an Association |
| `tnAPI` | an `APIFunction` over `tnSolve`, with parameters `expr`, `goal` and `dims` as the string `"a=2,b=3"` |
| `tnTrainAll[]` | generates the data, trains the three nets, writes the log notebook |
| `tnAccuracy[n]` | measures the nets against the oracles on n fresh examples |
| `tnSaveNets[]`, `tnLoadNets[]` | the nets to and from the Cloud; `tnSaveNets` writes nothing unless all three are trained |
| `tnLogSave[]` | writes the log notebook |
| `tensorNetsSave[]` | the release: saves the public functions and the bundle |
| `tnLex[s]` | the lexer: tokens and atoms |

### The examples, as returned by the kernel

Evaluated on 25 November 2026 (Wolfram 15.0) from `ff_tensornets_new.nb`, with no trained nets loaded:

| Input | Result |
|---|---|
| `tnParse["a ⊗ (b (+) c)"]` | `"String" -> "a ⨂ (b ⨁ c)"`, `"Prefix" -> {"P", "x1", "S", "x2", "x3"}`, `"Atoms" -> {"a", "b", "c"}`, `"Method" -> "Reference"` |
| `tnLex["(x1 + y)*z^2"]` | `"Tokens" -> {"(", "x1", "+", "x2", ")", "*", "x3", "^", "n2"}`, `"Atoms" -> {"x1", "y", "z"}` |
| `tnSimplify["(a⨁b)⨂c", "Expand"]` | `"Result" -> "a ⨂ c ⨁ b ⨂ c"`, one step `DistRight` at node 1, `"Verified" -> True` |
| `tnSimplify["a⨂c ⨁ b⨂c", "Factor"]` | `"Result" -> "(a ⨁ b) ⨂ c"`, one step `FactorRight` at node 1, `"Verified" -> True` |
| `tnSimplify["(a⨁0)⨂𝟙⨂b", "Simplify"]` | `"Result" -> "a ⨂ b"` in five steps: `AssocProd`, `DistRight`, `UnitProd`, `ZeroProd`, `ZeroSum` |
| `tnEvaluate["(V⨁W)^2", <\|"V" -> 2, "W" -> 3\|>]` | `"Exact" -> <\|"Dimension" -> 25, "Terms" -> 4\|>`, `"NetEstimate" -> Missing["NoNet"]` |
| `tnEvaluate["a⨂b", <\|"a" -> {{1,2},{3,4}}, "b" -> {{0,1},{1,0}}\|>]` | `"Value" -> {{0,1,0,2},{1,0,2,0},{0,3,0,4},{3,0,4,0}}`, the Kronecker product |
| `tnSolve["(a⨁b)⨂(c⨁d)", "Expand", <\|"a"->2, "b"->3, "c"->1, "d"->4\|>]` | `"result" -> "a ⨂ c ⨁ a ⨂ d ⨁ b ⨂ c ⨁ b ⨂ d"` in 4 steps, `"verified" -> True`, `"dimension" -> 25` |
| `tnSaveNets[]` | `Failure["NoNets", …]`: "No trained nets: run tnTrainAll[] first. Nothing was written." |

### Rules of the kernel

- **The nets decide; the exact engine executes.** Every rewrite is one of the eleven sound rules, so every result is mathematically correct, whatever the quality of the nets.
- **Every result is verified** by `tnEquivalentQ`: both sides are expanded into words (ordered ⨂-factors) and compared as multisets. ⨁ is commutative up to natural isomorphism; the order of ⨂ is never changed.
- **The nets are never trusted alone.** With `Method -> Automatic`, the exact result is returned whenever a net disagrees, and the disagreement is reported.
- **Nothing is printed, and nothing goes to the Cloud** unless you call `tnSaveNets[]` or `tensorNetsSave[]`. While the release saves, every symbol in the cell's context that does not belong to the project is blocked.


## 7. Python

tnParserNet itself runs in Wolfram only, for now. Our own Python code (`ccn_nets.nn`, `ccnlab.nn`) has dense layers but no attention yet, so the CCN Net Diagram Editor writes transformer nets for Wolfram only. The paths to Python:

| Path | Status |
|---|---|
| ONNX export from Wolfram 15 | a small dense net has been carried across this way; whether `AttentionLayer` with `"MultiHead"` exports cleanly is still to be tested |
| Attention in our own Python | planned, with its gradient check, as for the dense layers |
| The lexer, oracles and rule engine | small, and port directly |

The exact counterpart already exists: `python/tensor_parser.py` is the Python version of `tnLex` + `tnRefParse`, with the same grammar and the same operator spellings, and no dependencies.

```python
>>> import tensor_parser as T
>>> n = T.parse("a ⊗ (b (+) c)")
>>> T.to_str(n), T.to_prefix(n)
('a ⨂ (b ⨁ c)', 'TensorProduct[a, DirectSum[b, c]]')
>>> T.to_str(T.expand(T.parse("(V⨁W)^2")))
'V ⨂ V ⨁ V ⨂ W ⨁ W ⨂ V ⨁ W ⨂ W'
>>> T.dimension(T.parse("(V⨁W)^2"), {"V": 2, "W": 3})
25
>>> T.parse("(a⨁b")
ParseError: Expected RP, got 'end of input' at 4
```

| Function | Does | Wolfram counterpart |
|---|---|---|
| `tokenize(s)` | the tokens | `tnLex` |
| `parse(s)` | string → tree (`Sym`, `DirectSum`, `TensorProd`); raises `ParseError` | `tnRefParse` |
| `to_str(node)`, `to_prefix(node)` | back to a string; the S-expression form | the `"String"` of `tnParse` |
| `expand(node)` | ⨂ distributed over ⨁ | `tnSimplify[s, "Expand"]`'s result |
| `dimension(node, dims)` | the dimension | `tnDimension` |
| `evaluate(node, env)` | ⨂ → Kronecker product, ⨁ → block-diagonal or concatenation (NumPy) | `tnNumeric` |

`tensor_parser.parse` is the oracle tnParserNet is checked against, so a Python program can already parse every expression the net is meant to: the net adds speed and robustness to unusual spellings, never correctness.


## 8. Training it yourself

| Way | How |
|---|---|
| The whole system | `tnTrainAll[]` in `ff_tensornets_new.nb`: generates the data from the oracles, trains the three nets and writes `ff_tensornets_log.nb`. Expect an hour or more on a CPU at the defaults |
| Measuring | `tnAccuracy[n]`: the nets against the oracles on n fresh examples |
| From the diagram | open the CCN Net Diagram Editor, Examples ▸ tnParserNet, change it and generate a Wolfram notebook. `nets/tnparsernet/train/tnParserNetFromDiagram.nb` is that notebook: the diagram, the parts, the net, Wolfram's count (1,422,870) and a `NetTrain` cell for your own `trainData` (Adam, learning rate 0.005, L2 10⁻⁴, batches of 64, 12 rounds, seed 1234) |

**The data.** Random trees, each written in a random spelling: random glyphs for ⨁, ⨂ and powers, random spacing and redundant parentheses. The target is the tree's prefix form, from the oracle.

## 9. Results

| Run | Result |
|---|---|
| The test notebook (22 of 22 pass) | the lexer and the reference parser on 2,000 random spellings; the fall-back to the oracle; the net's wiring |
| A longer development run, d = 64, about six minutes on a CPU | 83 % of shallow trees rebuilt exactly |
| Full size (d = 128) | **not yet trained** |

The small test nets (d = 32) check the wiring, not the quality.

## 10. A complete worked session

```wolfram
(* evaluate the one cell of ff_tensornets_new.nb first *)
p = tnParse["(a⨁b)⨂c"]
(* <|"Tree" -> tnP[tnS[tnX[1], tnX[2]], tnX[3]], "Atoms" -> {"a", "b", "c"}, "String" -> "(a ⨁ b) ⨂ c",
     "Prefix" -> {"P", "S", "x1", "x2", "x3"}, "Method" -> "Reference", "NetAgreesWithReference" -> Missing["NoNet"]|> *)

e = tnSimplify["(a⨁b)⨂c", "Expand"]
(* <|"Input" -> "(a ⨁ b) ⨂ c", "Goal" -> "Expand", "Result" -> "a ⨂ c ⨁ b ⨂ c",
     "Tree" -> tnS[tnP[tnX[1], tnX[3]], tnP[tnX[2], tnX[3]]],
     "Steps" -> {<|"Rule" -> "DistRight", "Node" -> 1, "Result" -> "a ⨂ c ⨁ b ⨂ c"|>},
     "Method" -> "Oracle", "Complete" -> True, "Verified" -> True|> *)

f = tnSimplify[e["Result"], "Factor"]["Result"]            (* "(a ⨁ b) ⨂ c": back where we started *)

tnEvaluate["(V⨁W)^2", <|"V" -> 2, "W" -> 3|>]
(* <|"Exact" -> <|"Dimension" -> 25, "Terms" -> 4|>, "NetEstimate" -> Missing["NoNet"]|> *)

tnSolve["(a⨁b)⨂(c⨁d)", "Expand", <|"a" -> 2, "b" -> 3, "c" -> 1, "d" -> 4|>]["steps"]
(* {"DistRight @1 → a ⨂ (c ⨁ d) ⨁ b ⨂ (c ⨁ d)", "DistLeft @2 → a ⨂ c ⨁ a ⨂ d ⨁ b ⨂ (c ⨁ d)",
    "AssocSum @1 → a ⨂ c ⨁ a ⨂ d ⨁ b ⨂ (c ⨁ d)", "DistLeft @9 → a ⨂ c ⨁ a ⨂ d ⨁ b ⨂ c ⨁ b ⨂ d"} *)
```

For tnParserNet, the first call is the one that matters: `"Method" -> "Reference"` shows that the oracle answered, since no net is loaded. After `tnTrainAll[]`, the same call returns `"Method" -> "Net"` with `"NetAgreesWithReference" -> True` when the net is right.

## 11. Messages you may see

| Message | Meaning | What to do |
|---|---|---|
| `Failure["NoNets", …]` from `tnSaveNets[]` | no trained nets | run `tnTrainAll[]` first; nothing was written |
| `Failure["ParseError", …]`: "Expected ) at token 5, got EOF" | a parenthesis is not closed | close it |
| `"NetEstimate" -> Missing["NoNet"]`, `"NetAgreesWithReference" -> Missing["NoNet"]` | the nets are not trained or loaded; the exact result is given | `tnTrainAll[]` or `tnLoadNets[]` |
| `"Method" -> "Oracle"` or `"Reference"` | the exact engine answered | normal without nets, or when a net disagreed |
| `tnSolve`'s `"dimension"` comes back as an unevaluated `tnDimension[…, "a=2,b=3"]` | `dims` was given as a string | give an Association, `<\|"a" -> 2, "b" -> 3\|>`; the string form is for `tnAPI` |
| Python `ParseError: Expected RP, got 'end of input'` | as above, in Python | close the parenthesis |

## 12. Quick reference card

```
WOLFRAM (evaluate ff_tensornets_new.nb)            PYTHON (python/tensor_parser.py)
tnParse[s]   tnParse[s, Method -> "Net"]           T.parse(s)  T.to_str(n)  T.to_prefix(n)
tnSimplify[s, "Expand" | "Factor" | "Simplify"]    T.expand(n)
tnEvaluate[s, <|"a" -> 2|>]                         T.dimension(n, {"a": 2})  T.evaluate(n, env)
tnSolve[s, goal, dims]   tnAPI (dims "a=2,b=3")
tnTrainAll[]  tnAccuracy[n]  tnSaveNets[]  tnLoadNets[]  tnLogSave[]

tnParserNet  32 input tokens → 128 → 3 encoder + 3 decoder (4 heads, FF 512 GELU) → 22 prefix tokens
             1,422,870 numbers · oracle tnRefParse · decoding greedy + grammar
```

## 13. Files

| File | Holds |
|---|---|
| `ff_tensornets_new.nb` | the current code: every function in one cell |
| `ff_tensornets_test.nb`, `ff_tensornets_new_testlog.nb` | the tests (22 of 22 pass) and their latest results |
| `ff_tensornets_smallnets.nb` | small development nets (wiring checks only) |
| `DESIGN-tensornets.md` (and `.pdf`) | the design, with Appendix A explaining each term |
| `unit1_seq2seq.nb` … `unit4_build_your_own.nb`, `docs/units/` | the four units of learning the nets are built from |
| `nets/tnparsernet/` | the Net Lab folder: `net.json` (the diagram), `card.json`, `writeup.md`, `train/tnParserNetFromDiagram.nb`, `runs/`, this guide |
| `python/tensor_parser.py` | the Python parser and exact counterpart |

## 14. Credits, copyright and licence

By Dara O Shayda and Hikari Noor Hoshino (星野ヌール)
AI System Programming Laboratory

Follows Sutskever, Vinyals and Le (2014), Lample and Charton (2019), Vaswani et al. (2017) and Silver et al. (2016) for the policy and value nets, as cited in `DESIGN-tensornets.md`.

© Computational Classnotes · CC BY-NC-SA 4.0
