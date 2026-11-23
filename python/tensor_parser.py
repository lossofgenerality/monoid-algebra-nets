"""
tensor_parser.py — parse tensor-product / direct-sum expressions such as "(a⨁b)⨂c".

© Computational Classnotes · CC BY-NC-SA 4.0

Grammar (⨂ binds tighter than ⨁, like × over +; both are left-associative):

    expr   := term   ( SUM  term   )*
    term   := factor ( PROD factor )*
    factor := ATOM [ POW INT ] | "(" expr ")" [ POW INT ]

    SUM  : ⨁  ⊕  (+)  +
    PROD : ⨂  ⊗  (x)  *
    POW  : ^   e.g. a^3  means a⨂a⨂a,  also  a^⊗3 / a^⨂3
    ATOM : identifier (letters, digits, _, Greek, primes like a')

No dependencies. numpy is optional and only used by `evaluate`.
"""

from __future__ import annotations
from dataclasses import dataclass
import re

# ───────────────────────── AST ─────────────────────────

@dataclass(frozen=True)
class Sym:
    name: str
    def __str__(self): return self.name

@dataclass(frozen=True)
class DirectSum:
    args: tuple
    def __str__(self): return " ⨁ ".join(_wrap(a, DirectSum) for a in self.args)

@dataclass(frozen=True)
class TensorProd:
    args: tuple
    def __str__(self): return " ⨂ ".join(_wrap(a, TensorProd) for a in self.args)

def _wrap(node, parent):
    # parenthesize a sum inside a product
    if parent is TensorProd and isinstance(node, DirectSum):
        return f"({node})"
    return str(node)

def _flat(cls, items):
    """Build n-ary node, flattening nested same-type nodes (associativity)."""
    out = []
    for it in items:
        out.extend(it.args if isinstance(it, cls) else [it])
    return out[0] if len(out) == 1 else cls(tuple(out))

# ─────────────────────── Tokenizer ───────────────────────

_TOKEN_SPEC = [
    ("WS",    r"\s+"),
    ("SUM",   r"⨁|⊕|\(\+\)|\+"),
    ("PROD",  r"⨂|⊗|\([xX]\)|\*"),
    ("POW",   r"\^(?:⨂|⊗)?"),
    ("INT",   r"\d+"),
    ("LP",    r"\("),
    ("RP",    r"\)"),
    ("ATOM",  r"[^\W\d]\w*'*"),          # Unicode letters, digits, _, primes
]
_TOKEN_RE = re.compile("|".join(f"(?P<{n}>{p})" for n, p in _TOKEN_SPEC))

class ParseError(ValueError):
    pass

def tokenize(s: str):
    pos, toks = 0, []
    while pos < len(s):
        m = _TOKEN_RE.match(s, pos)
        if not m:
            raise ParseError(f"Unexpected character {s[pos]!r} at {pos}\n  {s}\n  {' '*pos}^")
        kind = m.lastgroup
        if kind != "WS":
            toks.append((kind, m.group(), pos))
        pos = m.end()
    toks.append(("EOF", "", pos))
    return toks

# ──────────────────────── Parser ────────────────────────

class Parser:
    def __init__(self, text: str):
        self.text = text
        self.toks = tokenize(text)
        self.i = 0

    def peek(self): return self.toks[self.i]

    def eat(self, kind):
        tok = self.peek()
        if tok[0] != kind:
            p = tok[2]
            got = tok[1] or "end of input"
            raise ParseError(f"Expected {kind}, got {got!r} at {p}\n  {self.text}\n  {' '*p}^")
        self.i += 1
        return tok

    def parse(self):
        node = self.expr()
        self.eat("EOF")
        return node

    def expr(self):
        items = [self.term()]
        while self.peek()[0] == "SUM":
            self.eat("SUM"); items.append(self.term())
        return _flat(DirectSum, items)

    def term(self):
        items = [self.factor()]
        while self.peek()[0] == "PROD":
            self.eat("PROD"); items.append(self.factor())
        return _flat(TensorProd, items)

    def factor(self):
        if self.peek()[0] == "LP":
            self.eat("LP"); node = self.expr(); self.eat("RP")
        else:
            node = Sym(self.eat("ATOM")[1])
        if self.peek()[0] == "POW":           # tensor power a^n = a⨂…⨂a
            self.eat("POW")
            n = int(self.eat("INT")[1])
            if n < 1:
                raise ParseError("Tensor power must be ≥ 1")
            node = _flat(TensorProd, [node] * n)
        return node

MAX_DEPTH = 10_000   # supported nesting depth of parentheses

def deep(fn):
    """Run a recursive function with a raised recursion limit on a thread with a
    large stack, so nesting up to MAX_DEPTH works without crashing the interpreter."""
    import functools, sys, threading
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if getattr(_tls, "active", False):          # already inside a deep call
            return fn(*args, **kwargs)
        result, error = [], []
        def run():
            _tls.active = True
            old = sys.getrecursionlimit()
            sys.setrecursionlimit(max(old, MAX_DEPTH * 5 + 1000))
            try:
                result.append(fn(*args, **kwargs))
            except BaseException as e:
                error.append(e)
            finally:
                sys.setrecursionlimit(old)
        old_size = threading.stack_size()
        threading.stack_size(512 * 1024 * 1024)     # 512 MB virtual; only touched pages use RAM
        try:
            th = threading.Thread(target=run); th.start(); th.join()
        finally:
            threading.stack_size(old_size)
        if error:
            if isinstance(error[0], RecursionError):
                raise ParseError(f"Expression nested deeper than MAX_DEPTH={MAX_DEPTH}") from error[0]
            raise error[0]
        return result[0]
    return wrapper

import threading as _threading
_tls = _threading.local()

@deep
def parse(text: str):
    return Parser(text).parse()

@deep
def to_str(node):
    return str(node)

# ─────────────────────── Transforms ───────────────────────

@deep
def expand(node):
    """Distribute ⨂ over ⨁:  (a⨁b)⨂c  →  a⨂c ⨁ b⨂c   (order preserved; ⨂ is not commutative)."""
    if isinstance(node, Sym):
        return node
    if isinstance(node, DirectSum):
        return _flat(DirectSum, [expand(a) for a in node.args])
    # TensorProd: cartesian product of each factor's summands
    terms = [[]]
    for a in node.args:
        e = expand(a)
        summands = e.args if isinstance(e, DirectSum) else (e,)
        terms = [t + [s] for t in terms for s in summands]
    return _flat(DirectSum, [_flat(TensorProd, t) for t in terms])

@deep
def dimension(node, dims: dict):
    """dim(a⨁b) = dim a + dim b ;  dim(a⨂b) = dim a · dim b"""
    if isinstance(node, Sym):
        return dims[node.name]
    vals = [dimension(a, dims) for a in node.args]
    if isinstance(node, DirectSum):
        return sum(vals)
    out = 1
    for v in vals: out *= v
    return out

@deep
def to_prefix(node):
    """S-expression form, handy for debugging or sending to another system."""
    if isinstance(node, Sym): return node.name
    head = "DirectSum" if isinstance(node, DirectSum) else "TensorProduct"
    return f"{head}[{', '.join(to_prefix(a) for a in node.args)}]"

@deep
def evaluate(node, env: dict):
    """Numeric evaluation with numpy: ⨂ → Kronecker product, ⨁ → block-diagonal
    (for matrices) or concatenation (for 1-D vectors)."""
    import numpy as np
    from functools import reduce
    if isinstance(node, Sym):
        return np.asarray(env[node.name])
    vals = [evaluate(a, env) for a in node.args]
    if isinstance(node, TensorProd):
        return reduce(np.kron, vals)
    if all(v.ndim == 1 for v in vals):
        return np.concatenate(vals)
    mats = [np.atleast_2d(v) for v in vals]
    r = sum(m.shape[0] for m in mats); c = sum(m.shape[1] for m in mats)
    out = np.zeros((r, c), dtype=np.result_type(*mats)); i = j = 0
    for m in mats:
        out[i:i+m.shape[0], j:j+m.shape[1]] = m
        i += m.shape[0]; j += m.shape[1]
    return out

# ───────────────────────── Demo ─────────────────────────

if __name__ == "__main__":
    for s in ["(a⨁b)⨂c", "a⨂b⨁c", "a ⊗ (b ⊕ c) ⊗ d", "(a(+)b)(x)(c(+)d)", "(V⨁W)^2", "ψ'⨂φ"]:
        t = parse(s)
        print(f"{s:22} → {str(t):22} | {to_prefix(t)}")
        print(f"{'':22}   expanded: {expand(t)}")

    t = parse("(a⨁b)⨂c")
    print("\ndim with a=2,b=3,c=4:", dimension(t, {"a": 2, "b": 3, "c": 4}))   # (2+3)*4 = 20

    try:
        import numpy as np
        M = evaluate(t, {"a": np.eye(2), "b": [[0, 1], [1, 0]], "c": np.diag([1, -1])})
        print("numeric shape:", M.shape)
    except ImportError:
        pass

    try:
        parse("(a⨁b⨂c")
    except ParseError as e:
        print("\nError example:\n" + str(e))
