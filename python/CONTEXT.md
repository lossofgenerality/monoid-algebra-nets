# CONTEXT — tensor-parser

A small pure-Python parser for **tensor-product (⨂)** and **direct-sum (⨁)** expressions written as strings, e.g. `"(a⨁b)⨂c"`. No Wolfram; no required dependencies (numpy optional).

This is a standalone project. It is related in subject to *DistributeNet-Wolfram* (a net that learns `(a⊕b)⊗c → (a⊗c)⊕(b⊗c)`), but is kept separate.

---

## Status (as of 2026-11-23)

- `tensor_parser.py` — working, tested. Single file, ~250 lines.
- Started 2026-11-14; nesting depth support raised to 10,000 the same day.
- Not yet on GitHub (account: `lossofgenerality`).

## Files

| File | Purpose |
|---|---|
| `tensor_parser.py` | Tokenizer, parser, AST, transforms, demo (`python3 tensor_parser.py`) |
| `CONTEXT.md` | This file — what the project is, how it works, decisions, limits |

## Grammar

`⨂` binds tighter than `⨁` (like × over +); both left-associative.

```
expr   := term   ( SUM  term   )*
term   := factor ( PROD factor )*
factor := ATOM [ POW INT ] | "(" expr ")" [ POW INT ]
```

| Token | Accepted forms |
|---|---|
| SUM  | `⨁` `⊕` `(+)` `+` |
| PROD | `⨂` `⊗` `(x)` `(X)` `*` |
| POW  | `^`, `^⨂`, `^⊗` — `a^3` = `a⨂a⨂a` (n ≥ 1) |
| ATOM | Unicode identifier with optional primes: `a`, `V1`, `ψ'`, `φ_2` |

## AST

Frozen dataclasses: `Sym(name)`, `DirectSum(args)`, `TensorProd(args)`.
Chains of the same operator are flattened into one n-ary node (`a⨂b⨂c` → one `TensorProd` with 3 args).

## Public API

| Function | Does |
|---|---|
| `parse(text)` | String → AST. Raises `ParseError` with position and caret |
| `to_str(node)` | Pretty-print with ⨁/⨂ (parenthesizes sums inside products) |
| `to_prefix(node)` | `TensorProduct[DirectSum[a, b], c]` form |
| `expand(node)` | Distributes ⨂ over ⨁, order preserved: `(a⨁b)⨂c → a⨂c ⨁ b⨂c` |
| `dimension(node, dims)` | dim adds under ⨁, multiplies under ⨂ |
| `evaluate(node, env)` | numpy: ⨂ → `np.kron`; ⨁ → block-diagonal (matrices) or concatenate (1-D vectors) |

```python
import tensor_parser as tp
t = tp.parse("(a⨁b)⨂c")
tp.to_prefix(t)                          # TensorProduct[DirectSum[a, b], c]
tp.to_str(tp.expand(t))                  # a ⨂ c ⨁ b ⨂ c
tp.dimension(t, {"a": 2, "b": 3, "c": 4})  # 20
```

## Design decisions

- **Hand-written recursive-descent parser** (precedence climbing) rather than Lark/pyparsing: transparent, no dependencies, easy to integrate.
- **⨂ treated as non-commutative**: `expand` never reorders factors.
- **Deep nesting**: recursive functions are wrapped by `@deep`, which runs them on a thread with a 512 MB (virtual) stack and a raised recursion limit. `MAX_DEPTH = 10_000` at the top of the file controls it.

## Tested limits

| Case | Result |
|---|---|
| Flat chain, 200,000 factors | parses in ~1 s (linear time) |
| Nesting 3,000 and 10,000 levels | all functions pass, < 0.1 s |
| Alternating ⨁/⨂ nesting, 3,000 levels | passes |
| Original version (before `@deep`) | failed between 300 and 400 levels |

## Known caveats

- `expand` output grows exponentially: `(a⨁b)` taken 20 times → 2²⁰ = 1,048,576 terms. This is the maths, not a bug.
- `evaluate` multiplies matrix sizes under ⨂ (ten 2×2 factors → 1024×1024). Check `dimension` first.
- Use `to_str(node)`, not `str(node)`, on trees deeper than ~1,000 levels — plain `str()` is not wrapped by `@deep`.
- A test at 10,001 levels did not raise the depth `ParseError`, because the actual recursion limit set is higher than `MAX_DEPTH`; the limit is soft.

## Alternatives considered

- **Lark** — best next step if the syntax grows (scalars, daggers, indices).
- **pyparsing** — `infix_notation` handles precedence in a few lines; slower, harder to debug.
- **SymPy** — `sympy.physics.quantum.TensorProduct` can do the algebra but has no direct sum; this parser would still be needed to build the SymPy objects.

---

© Computational Classnotes · CC BY-NC-SA 4.0
