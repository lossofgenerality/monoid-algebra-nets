---
title: "owNet: User Guide"
subtitle: "A small transformer that expands ⨂/⨁ expressions, checked by an exact oracle, in Wolfram, with its Python counterparts"
author: "Computational Classnotes"
date: "25 November 2026"
---

# owNet: User Guide

*A small transformer that expands ⨂/⨁ expressions, checked by an exact oracle, in Wolfram, with its Python counterparts*

Computational Classnotes · 25 November 2026

**Companion to:** the TensorNets units of learning (Unit 4 · Building Our Own Net) and the CCN Net Lab catalogue (https://lossofgenerality.github.io/ccn-nets/tensor-algebras/ownet/)

**Files covered:** `unit4_build_your_own.nb`, `unit4_build_your_own_nets.nb`, `docs/units/04-build-your-own.md`, `nets/ownet/train/owNetFromDiagram.nb`, `python/tensor_parser.py`

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

Given an expression such as (a⨁b)⨂c, owNet writes its expansion a⨂c ⨁ b⨂c. ⨂ is the tensor product and ⨁ the direct sum; ⨂ binds tighter than ⨁, distributes over it on both sides, and is not commutative.

owNet is an encoder–decoder transformer made small: d = 32, 4 heads, a feed-forward width of 128, and 2 encoder and 2 decoder layers. It has **59,950 numbers**: 59,722 learnt and 228 fixed. It reads infix tokens and writes prefix tokens. Every answer is checked by an exact oracle, so a wrong answer is never accepted.

It is the net of **Unit 4** of the TensorNets units, where it is built from the parts of Units 1–3. In the notebooks its symbol is `ownNet`; the catalogue calls it owNet. The superseded `lesson4_build_your_own.nb` holds an earlier net of the same design as `owNet`, decoded with `owBeam`.

## 2. Before you start

### What you need

| For | You need |
|---|---|
| Wolfram | Wolfram 14 or later (the numbers here are from Wolfram 15.0, CPU). Nothing else. |
| Python | Python 3.9 or later. NumPy only for `tensor_parser.evaluate`. |

### Opening it

1. Open `unit4_build_your_own.nb` and choose Evaluation ▸ Quit Kernel ▸ Local.
2. Either evaluate it from top to bottom (training in §6 takes about four minutes), or evaluate §1 (the toolkit) and then `unit4_build_your_own_nets.nb`, which restores the trained net as `ownNet` without training.

Nothing is written to the Wolfram Cloud. The notebook is saved with its outputs, so it can be read without running it.

## 3. Finding your way around

`unit4_build_your_own.nb` and its write-up `docs/units/04-build-your-own.md` have the same section numbers:

| § | Holds |
|---|---|
| 1 | the toolkit: vocabulary, sampler, oracle, the net's parts, training and decoding |
| 2 | the task: expand |
| 3 | the data: uniform binary trees |
| 4 | the oracle: words |
| 5 | the net, assembled from Unit 3 |
| 6 | training |
| 7 | decoding and checking |
| 8 | where the net goes wrong |
| 9 | saving the net |
| 10 | exercises, with answers |
| 11 | summary |

## 4. The mathematics

### The task

$$
(A\oplus B)\otimes C=A\otimes C\ \oplus\ B\otimes C,\qquad A\otimes(B\oplus C)=A\otimes B\ \oplus\ A\otimes C .
$$

### The oracle

An expression is a list of *words*, ordered ⨂-products of atoms:

$$
\mathrm{words}(x)=\{x\},\quad \mathrm{words}(A\oplus B)=\mathrm{words}(A)\cup\mathrm{words}(B),\quad
\mathrm{words}(A\otimes B)=\{\,uv : u\in\mathrm{words}(A),\ v\in\mathrm{words}(B)\,\}.
$$

Two expressions are equal when their lists of words agree up to order. For (a⨁b)⨂(c⨁d) the words are {ac, ad, bc, bd}.

### The net

Each token is embedded (d = 32), scaled by √d and given sinusoidal positions,

$$
\mathrm{PE}(p,2i)=\sin\!\left(p/10000^{2i/d}\right),\qquad \mathrm{PE}(p,2i+1)=\cos\!\left(p/10000^{2i/d}\right).
$$

Each attention head computes

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^{\mathsf T}}{\sqrt{d_k}}\right)V,\qquad d_k=d/h=8,
$$

with a causal mask in the decoder's self-attention. Every sub-layer is wrapped as LayerNorm(x + Sublayer(x)). The output layer shares its weights with the two embeddings.

### The loss

The target is label-smoothed with ε = 0.1 over the 10 tokens: 0.91 on the right token and 0.01 on each other. The loss is the cross-entropy against it, so its floor is the entropy of the smoothed target:

$$
-(0.91\ln 0.91+9\times0.01\ln0.01)=0.500 .
$$

### The decoding

Beam search with the length penalty ((5 + |y|)/6)^α, α = 0.6, optionally constrained by the prefix grammar: a counter of subtrees still owed allows `</s>` only when the tree is complete.

## 5. The configuration, part by part

The vocabulary has 10 tokens: `<s>`, `</s>`, a, b, c, d, ⨂, ⨁, (, ).

| Part | Numbers | Learnt? |
|---|---|---|
| one attention layer: 4 × (32² + 32) | 4,224 | yes |
| one feed-forward layer: 32·128 + 128 + 128·32 + 32 | 8,352 | yes |
| one LayerNorm: 2 × 32 | 64 | yes |
| encoder layer: attention + feed-forward + 2 LayerNorms | 12,704 | yes |
| decoder layer: 2 attentions + feed-forward + 3 LayerNorms | 16,992 | yes |
| 2 encoder + 2 decoder layers | 59,392 | yes |
| the shared embedding, 10 × 32 | 320 | yes |
| output biases | 10 | yes |
| label smoothing table, 10 × 10 | 100 | fixed |
| the two sinusoidal position layers | 128 | fixed |
| **Total** | **59,950** (59,722 learnt, 228 fixed) | |

Wolfram's `Information[NetInitialize[net], "ArraysTotalElementCount"]` returns 59,950 for the net in `nets/ownet/train/owNetFromDiagram.nb` (checked 25 November 2026).

| Setting | Value |
|---|---|
| Data | uniform binary trees with 1–5 operators over a, b, c, d; source ≤ 25 tokens, target ≤ 40 |
| Pairs | 12,000 training, 100 test; no test source in training |
| Optimiser | Adam, β₂ = 0.98, the warm-up schedule of Unit 3 with 400 warm-up steps |
| Batch, rounds | 64, 40 |
| Label smoothing | ε = 0.1 |
| Decoding | greedy; beam 4, α = 0.6; beam 4 + prefix grammar |

## 6. Using it in Wolfram

### Running

After §1 and `unit4_build_your_own_nets.nb`:

```wolfram
First@First@beamSearch[ownNet, Characters["(a⨁b)⨂c"], 4, True]
(* {"⨁", "⨂", "a", "c", "⨂", "b", "c"} *)
```

### The functions (all in §1)

| Function | Does |
|---|---|
| `beamSearch[net, src, width, grammar, α:0.6, maxLen:42]` | decodes a list of source tokens; returns {prefix tokens, score} pairs, best first. `grammar` True uses the prefix grammar |
| `parseT[tokens]` | prefix tokens → tree, or `$Failed` if they are not a tree |
| `infix[tree]` | tree → infix tokens with the fewest parentheses |
| `words[tree]` | the oracle's list of words |
| `expand[tree]` | the oracle's expansion |
| `equivalentQ[tree, tokens]` | True if the prefix tokens are a tree with the same words |
| `randomPrefix[n]`, `binaryTree[n]` | a uniform random tree with n operators (Unit 2, Algorithm 1) |
| `makePair[]` | one training pair |
| `model[s]`, `train[s, pairs]`, `trainRate[step, s]` | the net from settings s, its training, the warm-up schedule |
| `encoderLayerNet`, `decoderLayerNet`, `mhaNet`, `ffnNet`, `peNet` | the parts, from Unit 3 |
| `allowed[prefix, grammar]`, `need[prefix]` | the grammar: tokens allowed next, subtrees still owed |
| `saveNet[net, file]` | writes the net into its own notebook as `ownNet = Uncompress["…"];` |

### Reading the output

`beamSearch` returns a list of {tokens, score} pairs. `First@First@` takes the tokens of the best. The tokens are in prefix (Polish) form: `⨁ ⨂ a c ⨂ b c` is ⨁(⨂(a, c), ⨂(b, c)) = a⨂c ⨁ b⨂c.

### Examples (all returned by the kernel)

| Input | Result |
|---|---|
| `beamSearch[ownNet, Characters["(a⨁b)⨂c"], 4, True]`, best | `⨁ ⨂ a c ⨂ b c` |
| the net on a⨂(b⨁c) | `⨁ ⨂ a b ⨂ a c`, which the oracle accepts |
| `words` of (a⨁b)⨂(c⨁d) | {ac, ad, bc, bd}; its expansion has 15 prefix tokens from a source of 11 |
| `need` after `⨁ a`, after `⨁ a b` | 1, 0 |
| `equivalentQ` of a⨂c ⨁ b⨂c in either order; of c⨂a ⨁ c⨂b | True; False (the order of factors matters) |

**Try this.** Expand a⨂(b⨁c)⨂d by hand, then ask the net. The words are {abd, acd}.

### Rules of the kernel

- The net decides; the oracle checks. An answer that fails `equivalentQ` is never accepted.
- With the grammar on, every answer is a valid tree.
- The notebook prints nothing it does not compute, and writes nothing to the Cloud.

## 7. Python

owNet itself runs in Wolfram only, for now. Our own Python code (`ccn_nets.nn`, `ccnlab.nn`) has dense layers but no attention yet, so the CCN Net Diagram Editor writes transformer nets for Wolfram only. The paths to Python are:

| Path | Status |
|---|---|
| ONNX export from Wolfram 15 | a small dense net has been carried across this way; whether `AttentionLayer` with `"MultiHead"` exports cleanly is still to be tested |
| Attention in our own Python | planned, with its gradient check, as for the dense layers |

The **oracle** side already exists in Python: `python/tensor_parser.py` parses the same notation (and more spellings) and expands it, with no dependencies.

```python
>>> import tensor_parser as T
>>> n = T.parse("(a⨁b)⨂c")
>>> T.to_str(n), T.to_prefix(n), T.to_str(T.expand(n))
('(a ⨁ b) ⨂ c', 'TensorProduct[DirectSum[a, b], c]', 'a ⨂ c ⨁ b ⨂ c')
>>> T.to_str(T.expand(T.parse("a ⊗ (b (+) c)")))
'a ⨂ b ⨁ a ⨂ c'
```

| Function | Does |
|---|---|
| `parse(text)` | string → tree (`Sym`, `DirectSum`, `TensorProd`); raises `ParseError` |
| `to_str(node)`, `to_prefix(node)` | back to a string; the S-expression form |
| `expand(node)` | ⨂ distributed over ⨁: the oracle's expansion |
| `dimension(node, dims)` | the dimension, given each atom's |
| `evaluate(node, env)` | ⨂ → Kronecker product, ⨁ → block-diagonal or concatenation (NumPy) |

So a Python program can already check owNet's answers: an answer is right when `expand` of the source and of the answer have the same terms, in any order of ⨁ and the same order of ⨂.

## 8. Training it yourself

| Way | How |
|---|---|
| Unit 4 | evaluate `unit4_build_your_own.nb` from top to bottom; §6 trains (208 s in the recorded run), §9 saves |
| From the diagram | open the CCN Net Diagram Editor, Examples ▸ owNet, and generate a Wolfram notebook. `nets/ownet/train/owNetFromDiagram.nb` is that notebook: the diagram, the parts, the net, Wolfram's count (59,950) and a training cell for your own `trainData` |

## 9. Results

Recorded in Unit 4 (Wolfram 15.0, CPU; 100 test sources):

| Decoding | Valid prefix | Same tokens as the oracle | Top answer verified | Any in beam verified |
|---|---|---|---|---|
| greedy | 100 % | 86 % | 86 % | 86 % |
| beam 4, α = 0.6 | 100 % | 84 % | 84 % | 94 % |
| beam 4 + prefix grammar | 100 % | 87 % | 87 % | **97 %** |

| Training | Value |
|---|---|
| time | 208 s |
| loss, first and last round | 1.509, 0.508 |
| floor (label smoothing) | 0.5003 |

| Target length | Test sources | Top answer verified |
|---|---|---|
| 1–10 tokens | 66 | 89.4 % |
| 11–20 tokens | 31 | 87.1 % |
| 21–40 tokens | 3 | 33.3 % |

A single wrong token spoils a whole expansion, so long targets are harder. Only 3 test targets are longer than 20 tokens, so the last figure is uncertain.

## 10. A complete worked session

```wolfram
(* §1 of unit4_build_your_own.nb, then unit4_build_your_own_nets.nb *)
src = Characters["(a⨁b)⨂c"]                  (* {"(", "a", "⨁", "b", ")", "⨂", "c"} *)
hyps = beamSearch[ownNet, src, 4, True];      (* up to 4 {tokens, score}, best first *)
best = First@First@hyps                       (* {"⨁", "⨂", "a", "c", "⨂", "b", "c"} *)
tree = parseT[best]                           (* {"⨁", {"⨂", "a", "c"}, {"⨂", "b", "c"}} *)
equivalentQ[parseT[{"⨂", "⨁", "a", "b", "c"}], best]   (* True: accepted *)
```

The same check in Python:

```python
import tensor_parser as T
terms = lambda s: sorted(map(str, T.expand(T.parse(s)).args))   # the words, order of ⨁ free
terms("(a⨁b)⨂c") == terms("b⨂c⨁a⨂c")      # True:  accepted
terms("(a⨁b)⨂c") == terms("c⨂a⨁c⨂b")      # False: the order of ⨂ matters
```

## 11. Messages you may see

| Message | Meaning | What to do |
|---|---|---|
| `ownNet` returned unevaluated | the net is not loaded | evaluate §1, then `unit4_build_your_own_nets.nb` |
| `parseT` gives `$Failed` | the tokens are not a complete prefix tree | decode with `grammar` True |
| `equivalentQ` gives False | the answer is wrong; take the next in the beam | use a beam of 4 |
| Python `ParseError: Expected RP, got 'end of input'` | a parenthesis is not closed | close it |

## 12. Quick reference card

```
WOLFRAM (unit4_build_your_own.nb §1, then unit4_build_your_own_nets.nb)
beamSearch[ownNet, Characters["(a⨁b)⨂c"], 4, True]   best first {tokens, score}
parseT[tokens]  infix[tree]  words[tree]  expand[tree]  equivalentQ[tree, tokens]

PYTHON (python/tensor_parser.py, the oracle side)
T.parse(s)  T.expand(n)  T.to_str(n)  T.to_prefix(n)  T.dimension(n, dims)  T.evaluate(n, env)

NET  10 tokens; d 32, 4 heads, FF 128 ReLU; 2 + 2 layers; shared embedding
     59,950 numbers (59,722 learnt, 228 fixed); label smoothing 0.1; Adam, warm-up 400
     97 % of unseen sources solved (beam 4 + grammar + oracle)
```

## 13. Files

| File | Holds |
|---|---|
| `unit4_build_your_own.nb` | Unit 4: toolkit, data, oracle, net, training, decoding, results |
| `unit4_build_your_own_nets.nb` | the trained net, `ownNet = Uncompress["…"];` |
| `docs/units/04-build-your-own.{html,pdf,md}` | the write-up of Unit 4 |
| `nets/ownet/` | the Net Lab folder: `net.json` (the diagram), `card.json`, `writeup.md`, `train/owNetFromDiagram.nb`, `runs/`, this guide |
| `python/tensor_parser.py` | the Python parser and oracle |

## 14. Credits, copyright and licence

By Dara O Shayda and Hikari Noor Hoshino (星野ヌール)
AI System Programming Group

Follows Sutskever, Vinyals and Le (2014), Lample and Charton (2019) and Vaswani et al. (2017), through Units 1–3.

© Computational Classnotes · CC BY-NC-SA 4.0
