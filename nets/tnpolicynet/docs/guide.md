---
title: "tnPolicyNet: User Guide"
subtitle: "The TensorNets rewrite policy: a transformer that chooses which sound rule to apply, and where, to expand, factor or simplify ⨂/⨁ expressions"
author: "Computational Classnotes"
date: "25 November 2026"
---

# tnPolicyNet: User Guide

*The TensorNets rewrite policy: a transformer that chooses which sound rule to apply, and where, to expand, factor or simplify ⨂/⨁ expressions*

Computational Classnotes · 25 November 2026

**Companion to:** TensorNets (`DESIGN-tensornets.md`), the TensorNets units of learning and the CCN Net Lab catalogue (https://lossofgenerality.github.io/ccn-nets/tensor-algebras/tnpolicynet/)

**Files covered:** `ff_tensornets_new.nb`, `ff_tensornets_test.nb`, `nets/tnpolicynet/train/tnPolicyNetFromDiagram.nb`, `python/tensor_parser.py`

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

tnPolicyNet simplifies. Given a goal and an expression in prefix form, it chooses **which rule** to apply and **at which node**. The exact engine applies the rule. Because every rule is sound and the engine only offers legal moves, every result is correct, whatever the quality of the net.

It is a transformer encoder with d = 128, 4 heads, a feed-forward width of 512 (GELU) and 3 layers, with two heads on top: **611,468 numbers**, all learnt. Its oracle is `tnOracleStep`: the first node in preorder at which a goal rule applies, and there the rule highest in the goal's order.

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
| Understand what tnPolicyNet computes | §4 and §5 |
| Work in Python | §7 |
| Train it | §8 |
| Learn how the parts are made | Units 1–4 in `docs/units/`, and Appendix A of `DESIGN-tensornets.md` |
| See or change the design | the CCN Net Diagram Editor, Examples ▸ tnPolicyNet |

## 4. The mathematics

### Attention and the blocks

Each head computes

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^{\mathsf T}}{\sqrt{d_k}}\right)V,\qquad d_k=d/h=32,
$$

and the four heads are joined and projected back to d = 128. Every sub-layer is wrapped as LayerNorm(x + Sublayer(x)). The feed-forward layer is W₂ GELU(W₁x + b₁) + b₂, applied at every position. Tokens are embedded in 128 dimensions and a learnt position embedding (up to 96 positions) is added.

### The rules

All eleven are exact and sound:

| Rule | Rewrites |
|---|---|
| `ZeroSum` | 0 ⨁ A → A, A ⨁ 0 → A |
| `ZeroProd` | 0 ⨂ A → 0, A ⨂ 0 → 0 |
| `UnitProd` | 𝟙 ⨂ A → A, A ⨂ 𝟙 → A |
| `PowerExpand` | A^2 → A ⨂ A; A^k → A ⨂ A^(k−1) for k > 2 |
| `DistRight` | (A ⨁ B) ⨂ C → A⨂C ⨁ B⨂C |
| `DistLeft` | A ⨂ (B ⨁ C) → A⨂B ⨁ A⨂C |
| `AssocSum` | (A ⨁ B) ⨁ C → A ⨁ (B ⨁ C) |
| `AssocProd` | (A ⨂ B) ⨂ C → A ⨂ (B ⨂ C) |
| `FactorLeft` | A⨂B ⨁ A⨂C → A ⨂ (B ⨁ C), also at the head of a longer sum |
| `FactorRight` | A⨂C ⨁ B⨂C → (A ⨁ B) ⨂ C, also at the head of a longer sum |
| `PowerFold` | A ⨂ A → A^2; A ⨂ A^k → A^(k+1); the same at the head of a longer product |

### The goals

- `"Expand"`: distribute ⨂ over ⨁, remove 0 and 𝟙, expand powers, right-associate. Rules, in the oracle's order: ZeroSum, ZeroProd, UnitProd, PowerExpand, DistRight, DistLeft, AssocSum, AssocProd.
- `"Factor"`: the reverse: take common factors out and fold powers. Rules: ZeroSum, ZeroProd, UnitProd, AssocSum, AssocProd, FactorLeft, FactorRight, PowerFold.
- `"Simplify"`: Expand, then Factor, keeping the result only if it is no longer than the input.

### The two heads

The goal token (`<Expand>` or `<Factor>`) is put before the prefix sequence. After the encoder, the **rule head** reads the goal position and gives a softmax over the 11 rules; the **node head** scores every position and gives a softmax over the nodes. Training minimises the two cross-entropies against the oracle's step:

$$
L=-\log p_{\text{rule}}(r^\ast)-\log p_{\text{node}}(k^\ast).
$$

### Choosing a move

The engine lists every **legal** (rule, node) pair (`tnLegalMoves`), and the pair with the largest p_rule(r) · p_node(k) is taken (`tnPolicyStep`). An illegal step can never be taken, and Expand always terminates in its normal form.

### Verification

`tnEquivalentQ` expands both sides into words and compares them as multisets. ⨁ is commutative up to natural isomorphism; the order of ⨂ is never changed.

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

### tnPolicyNet

| Part | Numbers |
|---|---|
| embedding: 22 tokens × 128, plus positions 96 × 128 | 15,104 |
| encoder, 3 blocks | 594,816 |
| rule head: the goal position → 11 rules, 11 × 128 + 11 | 1,419 |
| node head: every position → 1 score, 128 + 1, softmax over positions | 129 |
| **Total** | **611,468** |

| Port | Holds |
|---|---|
| `Input` | the goal token and the prefix tokens (vocabulary 22) |
| `Rule`, `Node` | the oracle's step, for training |
| `RuleProb`, `NodeProb` | the two distributions |
| `LossRule`, `LossNode` | the two losses |

Wolfram's `Information[NetInitialize[net], "ArraysTotalElementCount"]` returns **611,468** for the net in `nets/tnpolicynet/train/tnPolicyNetFromDiagram.nb` (checked 25 November 2026).

## 6. Using it in Wolfram

### Running

```wolfram
tnSimplify["(a⨁b)⨂c", "Expand"]
tnSimplify["a⨂c ⨁ b⨂c", "Factor"]
tnSimplify["(a⨁0)⨂𝟙⨂b", "Simplify"]
```

### Reading the output

| Key | Holds |
|---|---|
| `"Result"` | the result as a string |
| `"Tree"` | the result's tree; atom slots number the atoms of the result in order of first appearance |
| `"Steps"` | each step: `"Rule"`, `"Node"` (its preorder position) and `"Result"` |
| `"Method"` | `"Net"` or `"Oracle"` |
| `"Complete"` | True when the goal's normal form was reached |
| `"Verified"` | True when `tnEquivalentQ` confirms the result |

**Try this.** `tnSimplify["(a⨁0)⨂𝟙⨂b", "Simplify"]` takes five steps to `"a ⨂ b"`: `AssocProd` at node 1, `DistRight` at 1, `UnitProd` at 4, `ZeroProd` at 5 and `ZeroSum` at 1.

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

tnPolicyNet itself runs in Wolfram only, for now. Our own Python code (`ccn_nets.nn`, `ccnlab.nn`) has dense layers but no attention yet, so the CCN Net Diagram Editor writes transformer nets for Wolfram only. The paths to Python:

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

The rule engine and `tnOracleStep` are not in Python yet; `tensor_parser.expand` gives Expand's normal form directly, without the steps, so it can check any Expand result of the policy.


## 8. Training it yourself

| Way | How |
|---|---|
| The whole system | `tnTrainAll[]` in `ff_tensornets_new.nb`: generates the data from the oracles, trains the three nets and writes `ff_tensornets_log.nb`. Expect an hour or more on a CPU at the defaults |
| Measuring | `tnAccuracy[n]`: the nets against the oracles on n fresh examples |
| From the diagram | open the CCN Net Diagram Editor, Examples ▸ tnPolicyNet, change it and generate a Wolfram notebook. `nets/tnpolicynet/train/tnPolicyNetFromDiagram.nb` is that notebook: the diagram, the parts, the net, Wolfram's count (611,468) and a `NetTrain` cell for your own `trainData` (Adam, learning rate 0.005, L2 10⁻⁴, batches of 64, 12 rounds, seed 1234) |

**The data.** Random expressions, each walked to its goal by the oracle; every step (goal, expression, rule, node) is one training example.

## 9. Results

| Run | Result |
|---|---|
| The test notebook (22 of 22 pass) | Expand and Factor soundness; the order of ⨂ kept; net-driven Expand always complete and verified; net-driven Factor and Simplify always verified; Factor never longer than its input |
| Full size (d = 128) | **not yet trained** |

Because moves are chosen only among legal ones and every result is verified, a weak policy can make a simplification longer to find, but never wrong.

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

For tnPolicyNet, read the `"Steps"`: each is a rule and a node chosen from the legal pairs. With no net loaded the oracle chose (`"Method" -> "Oracle"`); after `tnTrainAll[]` the net chooses, and the result is still verified.

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

tnPolicyNet  goal + 22 prefix tokens → 128 → 3 encoder blocks (4 heads, FF 512 GELU)
             → rule head (11) and node head (per position) · 611,468 numbers · oracle tnOracleStep
RULES  ZeroSum ZeroProd UnitProd PowerExpand DistRight DistLeft AssocSum AssocProd
       FactorLeft FactorRight PowerFold          GOALS  Expand  Factor  Simplify
```

## 13. Files

| File | Holds |
|---|---|
| `ff_tensornets_new.nb` | the current code: every function in one cell |
| `ff_tensornets_test.nb`, `ff_tensornets_new_testlog.nb` | the tests (22 of 22 pass) and their latest results |
| `ff_tensornets_smallnets.nb` | small development nets (wiring checks only) |
| `DESIGN-tensornets.md` (and `.pdf`) | the design, with Appendix A explaining each term |
| `unit1_seq2seq.nb` … `unit4_build_your_own.nb`, `docs/units/` | the four units of learning the nets are built from |
| `nets/tnpolicynet/` | the Net Lab folder: `net.json` (the diagram), `card.json`, `writeup.md`, `train/tnPolicyNetFromDiagram.nb`, `runs/`, this guide |
| `python/tensor_parser.py` | the Python parser and exact counterpart |

## 14. Credits, copyright and licence

By Dara O Shayda and Hikari Noor Hoshino (星野ヌール)
AI System Programming Group

Follows Sutskever, Vinyals and Le (2014), Lample and Charton (2019), Vaswani et al. (2017) and Silver et al. (2016) for the policy and value nets, as cited in `DESIGN-tensornets.md`.

© Computational Classnotes · CC BY-NC-SA 4.0
