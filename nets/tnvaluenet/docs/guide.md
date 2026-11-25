---
title: "tnValueNet: User Guide"
subtitle: "The TensorNets value net: a transformer that estimates an expression's dimension, number of terms and expansion cost, beside the exact values"
author: "Computational Classnotes"
date: "25 November 2026"
---

# tnValueNet: User Guide

*The TensorNets value net: a transformer that estimates an expression's dimension, number of terms and expansion cost, beside the exact values*

Computational Classnotes · 25 November 2026

**Companion to:** TensorNets (`DESIGN-tensornets.md`), the TensorNets units of learning and the CCN Net Lab catalogue (https://lossofgenerality.github.io/ccn-nets/tensor-algebras/tnvaluenet/)

**Files covered:** `ff_tensornets_new.nb`, `ff_tensornets_test.nb`, `nets/tnvaluenet/train/tnValueNetFromDiagram.nb`, `python/tensor_parser.py`

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

tnValueNet estimates three numbers about an expression, quickly: its dimension, its number of terms after expansion and the number of Expand steps. It can warn before an expansion that would grow exponentially, and later it can guide a search for shorter factorisations. **The exact values always come from the symbolic evaluator** and are reported beside the estimate.

It is a transformer encoder with d = 128, 4 heads, a feed-forward width of 512 (GELU) and 3 layers, pooled and followed by a small regression head: **626,819 numbers**, all learnt. Its oracles are `tnDimension`, `tnTermCount` and `tnNumeric`.

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
| Understand what tnValueNet computes | §4 and §5 |
| Work in Python | §7 |
| Train it | §8 |
| Learn how the parts are made | Units 1–4 in `docs/units/`, and Appendix A of `DESIGN-tensornets.md` |
| See or change the design | the CCN Net Diagram Editor, Examples ▸ tnValueNet |

## 4. The mathematics

### Attention and the blocks

Each head computes

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^{\mathsf T}}{\sqrt{d_k}}\right)V,\qquad d_k=d/h=32,
$$

and the four heads are joined and projected back to d = 128. Every sub-layer is wrapped as LayerNorm(x + Sublayer(x)). The feed-forward layer is W₂ GELU(W₁x + b₁) + b₂, applied at every position. Tokens are embedded in 128 dimensions and a learnt position embedding (up to 96 positions) is added.

### The input

The prefix tokens, plus, for each token, the log₂ dimension of its atom (0 for an operator). The second input goes through a learnt layer to 128 numbers per token and is added to the token embedding.

### The outputs

$$
\hat{\mathbf y}=\left(\log_2(1+\dim),\ \log_2(1+\text{terms}),\ \log_2(1+\text{Expand steps})\right).
$$

Logarithms keep the targets small although dimensions multiply: dim(A⨂B) = dim A · dim B and dim(A⨁B) = dim A + dim B, so (V⨁W)² with dim V = 2 and dim W = 3 has dimension 5² = 25 and 4 terms.

### Pooled regression

After the encoder the 128 numbers of every position are averaged (mean pooling), passed through a 128-unit GELU layer and a linear layer to the three outputs. The loss is the mean squared error against the exact values:

$$
L=\frac13\sum_{j=1}^{3}\left(\hat y_j-y_j\right)^2 .
$$

### Numeric evaluation is not the net's job

⨂ → Kronecker product and ⨁ → block-diagonal (or concatenation) are left to `tnNumeric`: a net would add nothing but error there.

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

### tnValueNet

| Part | Numbers |
|---|---|
| embedding: 20 tokens × 128, plus positions 96 × 128 | 14,848 |
| dimensions: 1 → 128 per token | 256 |
| add | 0 |
| encoder, 3 blocks | 594,816 |
| mean pool | 0 |
| dense: 128 → 128, GELU | 16,512 |
| regression: 128 → 3 | 387 |
| **Total** | **626,819** |

| Port | Holds |
|---|---|
| `Input` | the prefix tokens (vocabulary 20) |
| `Dims` | one number per token: its atom's log₂ dimension |
| `Target` | the three exact values, for training |
| `Estimate`, `Loss` | the three estimates; the mean squared error |

Wolfram's `Information[NetInitialize[net], "ArraysTotalElementCount"]` returns **626,819** for the net in `nets/tnvaluenet/train/tnValueNetFromDiagram.nb` (checked 25 November 2026).

## 6. Using it in Wolfram

### Running

```wolfram
tnEvaluate["(V⨁W)^2", <|"V" -> 2, "W" -> 3|>]
tnEvaluate["a⨂b", <|"a" -> {{1, 2}, {3, 4}}, "b" -> {{0, 1}, {1, 0}}|>]   (* numeric *)
```

### Reading the output

| Key | Holds |
|---|---|
| `"Exact"` | `"Dimension"` and `"Terms"`, from the symbolic evaluator |
| `"NetEstimate"` | the value net's three estimates, or `Missing["NoNet"]` |
| `"Value"` | with matrices or vectors: the numeric value |

**Try this.** `tnEvaluate["(V⨁W)^2", <|"V" -> 2, "W" -> 3|>]` gives `"Dimension" -> 25` and `"Terms" -> 4`. Its targets for the net are log₂ 26 ≈ 4.70 and log₂ 5 ≈ 2.32.

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

tnValueNet itself runs in Wolfram only, for now. Our own Python code (`ccn_nets.nn`, `ccnlab.nn`) has dense layers but no attention yet, so the CCN Net Diagram Editor writes transformer nets for Wolfram only. The paths to Python:

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

The exact values the net is trained on can already be computed in Python:

```python
>>> e = T.expand(T.parse("(V⨁W)^2"))
>>> T.dimension(e, {"V": 2, "W": 3}), len(e.args)        # dimension and number of terms
(25, 4)
>>> import numpy as np
>>> T.evaluate(T.parse("a⨂b"), {"a": np.array([[1, 2], [3, 4]]), "b": np.array([[0, 1], [1, 0]])})
array([[0, 1, 0, 2],
       [1, 0, 2, 0],
       [0, 3, 0, 4],
       [3, 0, 4, 0]])
```

The number of Expand steps needs the rule engine, which is not in Python yet.


## 8. Training it yourself

| Way | How |
|---|---|
| The whole system | `tnTrainAll[]` in `ff_tensornets_new.nb`: generates the data from the oracles, trains the three nets and writes `ff_tensornets_log.nb`. Expect an hour or more on a CPU at the defaults |
| Measuring | `tnAccuracy[n]`: the nets against the oracles on n fresh examples |
| From the diagram | open the CCN Net Diagram Editor, Examples ▸ tnValueNet, change it and generate a Wolfram notebook. `nets/tnvaluenet/train/tnValueNetFromDiagram.nb` is that notebook: the diagram, the parts, the net, Wolfram's count (626,819) and a `NetTrain` cell for your own `trainData` (Adam, learning rate 0.005, L2 10⁻⁴, batches of 64, 12 rounds, seed 1234) |

**The data.** Random expressions with random atom dimensions; the targets are the exact dimension, number of terms and Expand steps, from the oracles.

## 9. Results

| Run | Result |
|---|---|
| The test notebook (22 of 22 pass) | the value net gives an estimate beside the exact dimension; numeric evaluation; the JSON output |
| Full size (d = 128) | **not yet trained** |

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

For tnValueNet, read `tnEvaluate`: `"Exact"` is always there; `"NetEstimate"` is `Missing["NoNet"]` until the nets are trained, and then holds the three estimates beside it.

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

tnValueNet  20 prefix tokens + log2 dims → 128 → 3 encoder blocks (4 heads, FF 512 GELU)
            → mean pool → 128 GELU → 3 outputs · 626,819 numbers · loss MSE
OUTPUTS  log2(1+dimension)  log2(1+terms)  log2(1+Expand steps) · oracles tnDimension tnTermCount
```

## 13. Files

| File | Holds |
|---|---|
| `ff_tensornets_new.nb` | the current code: every function in one cell |
| `ff_tensornets_test.nb`, `ff_tensornets_new_testlog.nb` | the tests (22 of 22 pass) and their latest results |
| `ff_tensornets_smallnets.nb` | small development nets (wiring checks only) |
| `DESIGN-tensornets.md` (and `.pdf`) | the design, with Appendix A explaining each term |
| `unit1_seq2seq.nb` … `unit4_build_your_own.nb`, `docs/units/` | the four units of learning the nets are built from |
| `nets/tnvaluenet/` | the Net Lab folder: `net.json` (the diagram), `card.json`, `writeup.md`, `train/tnValueNetFromDiagram.nb`, `runs/`, this guide |
| `python/tensor_parser.py` | the Python parser and exact counterpart |

## 14. Credits, copyright and licence

By Dara O Shayda and Hikari Noor Hoshino (星野ヌール)
AI System Programming Group

Follows Sutskever, Vinyals and Le (2014), Lample and Charton (2019), Vaswani et al. (2017) and Silver et al. (2016) for the policy and value nets, as cited in `DESIGN-tensornets.md`.

© Computational Classnotes · CC BY-NC-SA 4.0
