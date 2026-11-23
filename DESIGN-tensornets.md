# DESIGN — ff_tensornets

*Updated 23 November 2026, 04:05 UTC · Neural nets that parse, evaluate and simplify tensor-algebra expressions in ⨂ (tensor product) and ⨁ (direct sum). Wolfram Language first; Python follows later.*

By Dara O Shayda and Hikari Noor Hoshino (星野ヌール), AI System Programmer

---

## 1. The design in one paragraph

The work is split into **three separate nets**, since the three tasks differ in kind:

- parsing is sequence-to-sequence translation;[^1]
- simplification is a sequence of decisions;
- evaluation is regression.

Each net is paired with an **exact symbolic counterpart**, which labels its training data and checks its answers. Neural nets have been shown to learn symbolic mathematics from generated data[^2]; here they are never trusted alone. The nets decide; the exact engine executes. As a result, every rewrite the system performs is mathematically sound, whatever the quality of the nets. Terms marked † are explained in detail, with examples, in **Appendix A**.

| Task | Net | Kind | Exact counterpart (oracle) |
|---|---|---|---|
| Parse | `tnParserNet` | transformer encoder–decoder†[^3], tokens → prefix tree | `tnRefParse` (recursive descent, as in `tensor_parser.py`[^4]) |
| Simplify | `tnPolicyNet` | transformer encoder, two heads: *which rule*, *at which node*†; a *policy* net[^5] | `tnOracleStep` (first applicable rule, preorder scan†) |
| Evaluate | `tnValueNet` | transformer encoder, pooled regression†; a *value* net[^5] | `tnDimension`, `tnTermCount`, `tnNumeric` |

## 2. Representation

- **Lexer** (`tnLex`, deterministic)†: identifiers become atom slots `x1 … x8` in order of first appearance, so the nets learn structure rather than names. Operator spellings (`⨁ ⊕ (+) +`, `⨂ ⊗ (x) *`, `^ ^⊗ ^⨂`) are kept as distinct tokens; recognising them is part of the parser's task. `0`/`𝟎` is the zero space, `1`/`𝟙` the unit space†.
- **Tree**: binary heads `tnS[a,b]` (⨁), `tnP[a,b]` (⨂), `tnE[n,a]` (tensor power), atoms `tnX[i]`, `tnZero`, `tnUnit`. Binary heads make the prefix form unambiguous; associativity is a rewrite rule.
- **Prefix (Polish) form**†[^6] — `(a⨁b)⨂c` → `P S x1 x2 x3`. It is the parser's output and the input of the other two nets. Its token *k* is node *k* of the tree in preorder, which is how the policy net points at a node.

## 3. Net 1 — the parser

- **Architecture**†: token + position embeddings; encoder of *L* self-attention blocks; decoder of *L* blocks (causal self-attention, cross-attention to the encoder, feed-forward), each with residual connection and layer normalisation; softmax over the output vocabulary. Defaults *d* = 128, 4 heads, *L* = 3.
- **Training**†: teacher forcing on random trees written in random spellings (random glyphs, spacing and redundant parentheses), target = the tree's prefix form.
- **Decoding:** greedy and **grammar-constrained**. This is a simple form of grammar-constrained generation[^7]. A counter of subtrees still owed forbids premature ends, and forbids atom slots beyond those in the input, so the output is always a well-formed tree.
- **Use:** `tnParse` runs the net and checks it against the reference parser. With `Method -> Automatic`, the reference result is returned whenever the two differ, and the disagreement is reported. With `Method -> "Net"`, the net's own answer is returned.

## 4. Net 2 — the rewrite policy

- **Rules** (all exact and sound; term rewriting in the usual sense[^8]): `ZeroSum`, `ZeroProd`, `UnitProd`, `PowerExpand`, `DistRight`, `DistLeft`, `AssocSum`, `AssocProd`, `FactorLeft`, `FactorRight`, `PowerFold`.
- **Goals:**
  - `"Expand"`: distribute ⨂ over ⨁, remove 0 and 𝟙, expand powers, right-associate;
  - `"Factor"`: the reverse — take common factors out and fold powers;
  - `"Simplify"`: Expand, then Factor, keeping the result only if it is no longer than the input.
- **Architecture**†: the goal token is prepended to the prefix sequence; encoder as above; the *rule head* reads the goal position, the *node head* scores every position.
- **Inference:** the exact engine lists every **legal** (rule, node) pair, and the net's probabilities choose among them. An illegal step can therefore never be taken, and Expand always terminates in its normal form.
- **Verification:** every result is checked by `tnEquivalentQ`. Both sides are expanded into words (ordered ⨂-factors) and compared as multisets. ⨁ is commutative up to natural isomorphism; ⨂ order is never changed.

## 5. Net 3 — the value net

- **Input:** prefix tokens, plus each atom's log₂ dimension.
- **Output:** log₂(1 + dimension), log₂(1 + number of terms after expansion), and log₂(1 + Expand steps) — a pooled regression†.
- **Purpose:** a fast cost estimate. For instance, it can warn before an expansion that would grow exponentially (the caveat in `CONTEXT.md`[^4]), and later it can guide a search for shorter factorisations. Exact values always come from the symbolic evaluator and are reported beside the estimate.
- **Numeric evaluation** (⨂ → Kronecker product[^9]; ⨁ → block-diagonal or concatenation) is left to `tnNumeric`: a net would add nothing but error there.

## 6. Public functions

| Function | Does |
|---|---|
| `tnTrainAll[]` | Generates data, trains the three nets, writes the log notebook |
| `tnParse[s]` | String → tree, prefix, canonical string, method used |
| `tnSimplify[s, goal]` | `"Expand"`, `"Factor"` or `"Simplify"`, with every step and a verification flag |
| `tnEvaluate[s, dims]` | Exact dimension and term count, and the net's estimate; matrices give a numeric value |
| `tnSolve[s, goal, dims]` | All of the above, JSON-ready |
| `tnAPI` | `APIFunction` over `tnSolve` (parameters `expr`, `goal`, `dims` as `"a=2,b=3"`) |
| `tnAccuracy[n]` | Measures the nets against the oracles on fresh data |
| `tnSaveNets[]`, `tnLoadNets[]` | Nets to and from `AI/freeform/ff/tensornets/nets/`; `tnSaveNets` writes nothing unless all three nets are trained |
| `tnLogSave[]` | Writes the log notebook |
| `tensorNetsSave[]` | The release: CloudSaves the public functions and the bundle |

Nothing is printed. Training progress (per round) and summaries go to `ff_tensornets_log.nb`, beside the evaluating notebook.

## 7. Notebooks and routine

All work is in `~/Desktop/Research/TensorNets`, in notebooks only, each holding one Input cell, following CCN's working practice[^10].

| Notebook | Holds |
|---|---|
| `ff_tensornets_new.nb` | The current code: every function in one cell |
| `ff_tensornets_test.nb` | The tests |
| `ff_tensornets_new_testlog.nb` | The latest test results |
| `ff_tensornets_smallnets.nb` | The small development nets (wiring checks only) |
| `ff_tensornets.nb`, `ff_tensornets_testlog.nb` | The earlier version and its test log, kept unchanged |

- **Evaluating the cell writes nothing to the Wolfram Cloud.** Cloud writes happen only when the owner calls them: `tensorNetsSave[]` (the release) and `tnSaveNets[]` (the nets).
- While `tensorNetsSave[]` saves (with `CloudSave`[^11]), every symbol in the cell's context that does not belong to this project is blocked. Values left in a shared kernel can therefore never be swept into the release.
- Full-size training (`tnTrainAll[]`) has not yet been done. On CPU, expect an hour or more for all three nets at the defaults.

## 8. Tested (23 November 2026, Wolfram 15.0, separate `wolframscript` process)

`ff_tensornets_test.nb` evaluates the cell of `ff_tensornets_new.nb` and writes `ff_tensornets_new_testlog.nb`. It writes nothing to the Cloud and deletes nothing. Result: **22 of 22 checks pass**. They cover:

- the cell makes no Cloud call;
- the lexer and parser (2,000 random spellings);
- Expand and Factor soundness;
- ⨂ order;
- numeric evaluation;
- the worked examples;
- the fall-back to the oracles;
- stray-symbol blocking in the release;
- the refusal of `tnSaveNets` without trained nets;
- the nets' wiring;
- the JSON output and `tnAPI`.

The small test nets (*d* = 32) do not measure quality. In a separate, longer run, a parser with *d* = 64 trained for about six minutes rebuilt 83 % of shallow trees exactly.

## 9. Path to Python

- Nets export to ONNX[^12] from Wolfram 15; a small net has already been carried across this way in the agent free-form language work. Whether `AttentionLayer` with `"MultiHead"` exports cleanly remains to be tested.
- The lexer, oracles and rule engine are small and port directly. `tensor_parser.py` is already the Python counterpart of `tnLex` + `tnRefParse`: same grammar, same operator spellings.

## 10. Open questions

1. Should ⨂ be treated as commutative up to isomorphism (as ⨁ is), which would let Factor find more common factors?
2. Should scalars, duals or daggers enter the grammar now, or after the first full training?
3. Should Factor become a search guided by the value net (beam search), rather than greedy imitation of the oracle?

---

## Appendix A — Terms explained, with examples

The running example throughout is the expression **`(a⨁b)⨂c`**.

### A.1 Lexer (`tnLex`, deterministic) and atom slots

**Definition.** A *lexer* cuts a string into *tokens*, the smallest meaningful pieces: names, operators, parentheses and numbers. `tnLex` is *deterministic*: it is ordinary code, not a net, and the same string always gives the same tokens.

Whenever it meets a name (an *identifier*), it replaces the name with an **atom slot**: `x1` for the first distinct name, `x2` for the second, and so on up to `x8`. The real names are kept in a separate list, so the result can be printed with them afterwards.

**Why.** The meaning of `(a⨁b)⨂c` does not depend on the letters chosen. `(V⨁W)⨂U` and `(ψ'⨁φ_2)⨂H` have exactly the same structure. With slots, the nets see one pattern instead of thousands of spellings of it, and so learn structure rather than names.

**Examples.**

| Input string | Tokens | Atoms |
|---|---|---|
| `(a⨁b)⨂c` | `( x1 ⨁ x2 ) ⨂ x3` | `{a, b, c}` |
| `(V⨁W)⨂U` | `( x1 ⨁ x2 ) ⨂ x3` | `{V, W, U}` |
| `a⨂b⨁a` | `x1 ⨂ x2 ⨁ x1` | `{a, b}` (a repeated name keeps its slot) |
| `(V ⊕ W)^2` | `( x1 ⊕ x2 ) ^ n2` | `{V, W}` |

Operator spellings are kept as they were typed (`⨁` and `⊕` remain different tokens), because learning that they mean the same thing is part of the parser's task. `(x)` is always read as a ⨂ spelling, never as a name, so the letters `x` and `X` are not used as names.

### A.2 `0`/`𝟎` is the zero space, `1`/`𝟙` the unit space

**Definition.**

- The **zero space**[^13] is the vector space with one element, the zero vector. Its dimension is 0. For every space V: V ⨁ 0 ≅ V, and V ⨂ 0 ≅ 0.
- The **unit space** is the ground field itself (for example ℝ or ℂ), a space of dimension 1. For every space V: V ⨂ 𝟙 ≅ V. It plays the part that the number 1 plays in multiplication.

**Spelling.** Either may be typed as a plain digit (`0`, `1`) or as the double-struck character (`𝟎`, `𝟙`). A digit directly after `^` is an exponent, not a space: `a^2` is a⨂a. Any other number is an error.

**Examples** (with `tnSimplify[…, "Simplify"]`):

| Input | Result | Rule that removed it |
|---|---|---|
| `a ⨁ 0` | `a` | `ZeroSum` |
| `(a⨁b) ⨂ 0` | `0` | `ZeroProd` |
| `a ⨂ 𝟙 ⨂ b` | `a ⨂ b` | `UnitProd` |
| `(V⊕W)⨂(V⊕W) ⨂ 𝟙 ⨁ 0` | `(V ⨁ W)^2` | `ZeroSum`, `UnitProd`; expansion; then `FactorLeft`, `FactorRight`, `PowerFold` |

In numeric evaluation, `𝟙` is the 1 × 1 matrix `{{1}}`.

### A.3 Prefix (Polish) form and preorder

**Definition.** In *prefix form*, each operator is written **before** its operands, so no parentheses are needed:
- `S` stands for ⨁ (two operands);
- `P` stands for ⨂ (two operands);
- `E2 … E9` stand for the tensor powers ^2 … ^9 (one operand).

Reading the tree in **preorder**[^14] means visiting a node first, then its left subtree, then its right subtree. Read this way, the nodes come out in exactly the order of the prefix tokens.

**Example.** `(a⨁b)⨂c` is the tree P(S(x1, x2), x3):

| Prefix token | `P` | `S` | `x1` | `x2` | `x3` |
|---|---|---|---|---|---|
| Node number (preorder) | 1 | 2 | 3 | 4 | 5 |
| Meaning | the whole ⨂ | the ⨁ inside | a | b | c |

A tensor power becomes, for example, `(V⨁W)^2` → `E2 S x1 x2`.

### A.4 Preorder scan (the oracle's choice)

**Definition.** To choose the next rewrite step, the oracle `tnOracleStep` walks the nodes **in preorder** (node 1, node 2, …). At the **first** node where any rule allowed by the goal applies, it applies the highest-priority such rule. The priority order for Expand is `ZeroSum, ZeroProd, UnitProd, PowerExpand, DistRight, DistLeft, AssocSum, AssocProd`. In effect, the oracle always works on the outermost, leftmost place that can change. That makes it fully predictable, which is what a net learning to imitate it needs.

**Example** (goal Expand, on `(a⨁b)⨂c`):
1. Node 1 is P(S(a, b), c). `DistRight` applies: (A⨁B)⨂C → A⨂C ⨁ B⨂C. Result: `a ⨂ c ⨁ b ⨂ c`.
2. The new tree is S(P(a, c), P(b, c)). No Expand rule applies at any node, so the scan ends: this is the normal form.

**Longer example** (goal Expand, on `(V⊕W)^2`): `PowerExpand` (node 1), `DistRight` (node 1), `DistLeft` (node 2), `AssocSum` (node 1), `DistLeft` (node 9). Result: `V ⨂ V ⨁ V ⨂ W ⨁ W ⨂ V ⨁ W ⨂ W`.

### A.5 Transformer encoder (and its parts)

The three nets are built from the same parts.

- **Token embedding.** Each token is mapped to a learned vector of *d* numbers (*d* = 128 by default). The map is a lookup table: one row per vocabulary entry, adjusted during training.
- **Position embedding.** A second learned vector is added according to the token's place in the sequence (1st, 2nd, …). Without it, the net could not tell `P S x1 x2 x3` from the same tokens in another order.
- **Self-attention**[^3]. Every position looks at every position, itself included. It computes how relevant each one is (scores, turned into weights that sum to 1), then takes the weighted mix of their vectors. *Multi-head* means this is done four times in parallel with different learned projections, so that different heads can follow different relations. For example, in `( x1 ⨁ x2 ) ⨂ x3`, one head can learn to pair each `)` with its `(`, while another links `⨂` with its operands.
- **Feed-forward.** A small two-layer network applied to each position separately (*d* → 4*d* → *d*, with a GELU non-linearity[^15]).
- **Residual connection**[^16]. Each part's input is added to its output (output = x + f(x)). The block therefore only has to learn a correction, which keeps deep nets trainable.
- **Layer normalisation**[^17]. After each residual addition, the *d* numbers at each position are rescaled to mean 0 and variance 1, then multiplied and shifted by learned amounts. This keeps the scale steady from block to block.
- **Encoder.** A stack of *L* blocks (*L* = 3 by default), each block being self-attention then feed-forward, both with residual connection and layer normalisation. It turns a sequence of tokens into a sequence of vectors, one per token, each informed by the whole sequence.

### A.6 The parser's architecture (encoder–decoder)

> *Token + position embeddings; encoder of L self-attention blocks; decoder of L blocks (causal self-attention, cross-attention to the encoder, feed-forward), each with residual connection and layer normalisation; softmax over the output vocabulary.*

- **Encoder:** as in A.5, reading the lexer's tokens. The input vocabulary has 32 entries: 13 operator and parenthesis spellings, the powers `n1 … n9`, the slots `x1 … x8`, and `0` and `1`.
- **Decoder:** it writes the prefix form one token at a time. Each of its *L* blocks has three parts:
  - **Causal self-attention:** like self-attention, but position *i* may look only at positions 1 … *i*, the tokens already written. It cannot see the future, which it will not have when it really generates.
  - **Cross-attention to the encoder**[^18] — the decoder's vectors ask questions of the encoder's vectors. This is how the output is tied to the input string.
  - **Feed-forward:** as in A.5.
- **Softmax over the output vocabulary:** at each step, the final vector is turned into a probability for each of the 22 output tokens (`S`, `P`, `E2 … E9`, `x1 … x8`, `0`, `1`, start `<s>` and end `</s>`). The most probable *permitted* token is written.

**Example** (input `( x1 ⨁ x2 ) ⨂ x3`). The decoder writes `P`, `S`, `x1`, `x2`, `x3`. The grammar constraint keeps a count of subtrees still owed:

| Written token | none yet | `P` | `S` | `x1` | `x2` | `x3` |
|---|---|---|---|---|---|---|
| Subtrees still owed | 1 | 2 | 3 | 2 | 1 | 0 → stop |

An operator owes two subtrees and fills one (+1); an atom fills one (−1). A tree is complete exactly when the count reaches 0, so the output is always a well-formed tree.

### A.7 Teacher forcing on random spellings

> *Teacher forcing on random trees written in random spellings (random glyphs, spacing and redundant parentheses), target = the tree's prefix form.*

**Random trees.** The generator draws trees at random: depth 1 to 5, one to five atoms, with occasional 0, 𝟙 and powers ^2 and ^3.

**Random spellings.** Each tree is written out as a string in a random way:
- each operator gets a random glyph (⨁ ⊕ (+) +, or ⨂ ⊗ (x) *, in proportions 5 : 3 : 1 : 1);
- spaces are random;
- each subexpression is put in unnecessary parentheses with probability 8 %;
- the atoms get random names, such as `a`, `V'`, `ψ_1` or `B2`.

So one tree is seen in many disguises, and all of them have the same target. For P(S(x1, x2), x3), for example:

```
(a⨁b)⨂c        (ψ (+) V_1)*B'        ((u⊕ w))⊗(H2)
```

**Target.** The tree's prefix form, framed by start and end: `<s> P S x1 x2 x3 </s>`.

**Teacher forcing**[^19]. During training, the decoder is not left to its own guesses. At every step it is shown the **correct** previous tokens, and is scored on predicting the next one:

| Decoder is given | `<s>` | `<s> P` | `<s> P S` | `<s> P S x1` | `<s> P S x1 x2` | `<s> P S x1 x2 x3` |
|---|---|---|---|---|---|---|
| Must predict | `P` | `S` | `x1` | `x2` | `x3` | `</s>` |

The loss is the cross-entropy summed over all positions. All six predictions are trained at once (the causal mask of A.6 keeps each one honest), which makes training fast and stable. The weights are adjusted by `NetTrain`[^20] with its default optimiser, Adam[^21]; the attention layers are Wolfram's `AttentionLayer`[^22].

### A.8 Two heads: *which rule*, *at which node*

> *The goal token is prepended to the prefix sequence; encoder as above; the rule head reads the goal position, the node head scores every position.*

**Input.** The goal is written as an extra first token. For Expand on `(a⨁b)⨂c`, the sequence is:

| Position | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Token | `<Expand>` | `P` | `S` | `x1` | `x2` | `x3` |
| Tree node | (the goal) | 1 | 2 | 3 | 4 | 5 |

The encoder (A.5) turns this into six vectors. A *head* is a small output layer on top of them.

- **Rule head (*which rule*):** reads only the vector at position 1, the goal token. Through self-attention, that vector has gathered information about the whole expression. The head gives one probability per rule (11 rules), for example `DistRight` 0.92, `AssocSum` 0.03, and so on.
- **Node head (*at which node*):** gives every position a score (a pointer over the input, as in pointer networks[^23]), and a softmax turns the scores into probabilities over the positions. Position *k* + 1 stands for tree node *k*: here position 2 (node 1, the outer ⨂) might get 0.95.

**Combining them with the legal-move mask.** The exact engine lists the moves that are actually possible. Here there is only one: (`DistRight`, node 1). Each legal move is scored as *p*(rule) × *p*(node), and the best one is applied by the exact engine. The net therefore steers, but can never make an illegal or unsound step.

### A.9 Pooled regression (the value net)

> *Transformer encoder, pooled regression.*

**Definition.** The encoder gives one vector per token.
- **Pooling** combines them into a **single** vector for the whole expression; here, by taking their average.
- **Regression** means the output is a real number, not a choice among classes. A small feed-forward head maps the pooled vector to three numbers, and training minimises the mean squared error against the exact values.
- Besides the tokens, the net receives each atom's log₂ dimension (0 for operators), added to the token's vector.

**Example** (`(a⨁b)⨂c` with dim a = 2, dim b = 3, dim c = 4).

| Prefix token | `P` | `S` | `x1` | `x2` | `x3` |
|---|---|---|---|---|---|
| log₂ dimension supplied | 0 | 0 | 1 | 1.585 | 2 |

| Quantity | Exact value | What the net is trained to output |
|---|---|---|
| Dimension, (2 + 3) · 4 | 20 | log₂(21) = 4.39 |
| Terms after expansion (a⨂c ⨁ b⨂c) | 2 | log₂(3) = 1.585 |
| Expand steps (one `DistRight`) | 1 | log₂(2) = 1 |

The net's outputs *y* are turned back into numbers by 2^*y* − 1 and rounded. The logarithms keep huge values (an expansion with a million terms) and small ones on the same footing. `tnEvaluate` always reports the exact values beside the estimate.

---

© Computational Classnotes · CC BY-NC-SA 4.0

---

## References

[^1]: I. Sutskever, O. Vinyals and Q. V. Le, "Sequence to Sequence Learning with Neural Networks", *NeurIPS 2014*. arXiv:1409.3215. — Sequence-to-sequence translation.
[^2]: G. Lample and F. Charton, "Deep Learning for Symbolic Mathematics", *ICLR 2020*. arXiv:1912.01412. — Transformers trained on generated expressions in prefix form.
[^3]: A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser and I. Polosukhin, "Attention Is All You Need", *Advances in Neural Information Processing Systems 30* (NeurIPS 2017). arXiv:1706.03762. — The transformer: self-attention, multi-head attention, encoder–decoder.
[^4]: Computational Classnotes, `tensor_parser.py` and its `CONTEXT.md`, Tensor Algebras project, 2026. — The pure-Python parser for the same notation.
[^5]: D. Silver et al., "Mastering the game of Go with deep neural networks and tree search", *Nature* 529, 2016, pp. 484–489. — The terms *policy network* (which move) and *value network* (how good, or how costly, a position is).
[^6]: J. Łukasiewicz, *Elementy logiki matematycznej* (Elements of Mathematical Logic), Warsaw, 1929; English translation, Pergamon Press, 1963. — Prefix ("Polish") notation.
[^7]: M. J. Kusner, B. Paige and J. M. Hernández-Lobato, "Grammar Variational Autoencoder", *ICML 2017*. arXiv:1703.01925. — Generating only grammatically valid sequences by masking.
[^8]: F. Baader and T. Nipkow, *Term Rewriting and All That*, Cambridge University Press, 1998. — Rewrite rules, termination and normal forms.
[^9]: C. F. Van Loan, "The ubiquitous Kronecker product", *Journal of Computational and Applied Mathematics* 123, 2000, pp. 85–100.
[^10]: Computational Classnotes, `ccn-hoshino-and-colleagues` (private repository): `common/working-practice.md`, `common/wolfram-cloud-build-process.md` and `projects/tensor-algebras.md`, 2026.
[^11]: Wolfram Research, `CloudSave`, Wolfram Language Documentation. https://reference.wolfram.com/language/ref/CloudSave.html
[^12]: ONNX, Open Neural Network Exchange. https://onnx.ai
[^13]: P. R. Halmos, *Finite-Dimensional Vector Spaces*, 2nd ed., Van Nostrand, 1958 (reprinted by Springer, 1974). — Direct sums, tensor products, the zero space and the ground field.
[^14]: D. E. Knuth, *The Art of Computer Programming, Vol. 1: Fundamental Algorithms*, 3rd ed., Addison-Wesley, 1997, §2.3.1 "Traversing Binary Trees". — Preorder.
[^15]: D. Hendrycks and K. Gimpel, "Gaussian Error Linear Units (GELUs)", 2016. arXiv:1606.08415.
[^16]: K. He, X. Zhang, S. Ren and J. Sun, "Deep Residual Learning for Image Recognition", *Proceedings of CVPR 2016*, pp. 770–778. arXiv:1512.03385. — Residual connections.
[^17]: J. L. Ba, J. R. Kiros and G. E. Hinton, "Layer Normalization", 2016. arXiv:1607.06450.
[^18]: D. Bahdanau, K. Cho and Y. Bengio, "Neural Machine Translation by Jointly Learning to Align and Translate", *ICLR 2015*. arXiv:1409.0473. — Attention from a decoder to an encoder (cross-attention).
[^19]: R. J. Williams and D. Zipser, "A Learning Algorithm for Continually Running Fully Recurrent Neural Networks", *Neural Computation* 1(2), 1989, pp. 270–280. — The origin of teacher forcing.
[^20]: Wolfram Research, `NetTrain`, Wolfram Language Documentation. https://reference.wolfram.com/language/ref/NetTrain.html
[^21]: D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization", *ICLR 2015*. arXiv:1412.6980.
[^22]: Wolfram Research, `AttentionLayer`, Wolfram Language Documentation. https://reference.wolfram.com/language/ref/AttentionLayer.html
[^23]: O. Vinyals, M. Fortunato and N. Jaitly, "Pointer Networks", *NeurIPS 2015*. arXiv:1506.03134. — An output that points at a position of the input.
