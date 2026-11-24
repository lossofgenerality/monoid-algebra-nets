"""python3 tools/units/build_glossary.py -> docs/glossary.html, docs/units/index.html, docs/net-diagrams.html and index.html
Every technical term of the units, defined where it is worked out. © Computational Classnotes · CC BY-NC-SA 4.0"""
import pathlib, html
ROOT = pathlib.Path(__file__).resolve().parents[2]
CSS = (ROOT / 'tools/units/pdf-style.css').read_text(encoding='utf-8')
SCREEN = '''body{max-width:860px;margin:0 auto;padding:28px 22px 60px;background:#fff}
nav.wtop{font-size:10pt;margin-bottom:14px;display:flex;gap:14px;flex-wrap:wrap} nav.wtop a{color:rgb(212,86,28)}
dt{font-weight:bold;margin-top:10px} dd{margin:2px 0 6px 18px} dd .ex{color:#555} dd a{font-size:9pt}
.unit{border:1px solid #c9d0db;border-radius:6px;padding:10px 14px;margin:10px 0} .unit h3{margin:0 0 4px}'''
U = {1: '01-seq2seq', 2: '02-symbolic-data', 3: '03-transformer', 4: '04-build-your-own'}
REPO = 'https://github.com/lossofgenerality/tensor-algebra-nets'
# The Net Diagrams editor lives in the CCN Net Lab and is published in ccn-nets; this site shows that live copy, so a change there appears here at once
EDITOR = 'https://lossofgenerality.github.io/ccn-nets/editor/net-diagram-editor.html?example=tnparser'   # opens on tnParserNet
EDITOR_README = 'https://lossofgenerality.github.io/ccn-nets/editor/guide.html'

# term: (definition, example or formula, unit, section)
TERMS = {
 'add & LayerNorm (residual connection)': ('Each sub-layer of a transformer is wrapped as LayerNorm(x + Sublayer(x)): its input is added back to its output, then normalised.', 'Lets the signal and its gradient pass straight through.', 3, 8),
 'arity': ('The number of arguments a symbol takes.', 'add and mul have arity 2, sin 1, a number 0.', 2, 3),
 'atom': ('A symbol that is not an operator; in Unit 4, a, b, c or d.', '', 4, 3),
 'attention': ('An average of values, weighted by how well each key matches a query: softmax(QKᵀ/√dₖ)V.', 'q = (1, 0), keys (1, 0) and (0, 1) give weights 0.670 and 0.330.', 3, 3),
 'backward generation (BWD)': ('Making integration data by drawing F and differentiating it; the pair is (F′, F).', 'F = x sin x gives the problem sin x + x cos x.', 2, 6),
 'batch': ('The group of training examples whose gradients are averaged for one update.', 'The paper of Unit 1 used 128 sentences.', 1, 7),
 'beam search': ('Decoding that keeps the B most likely partial outputs at every step, instead of only the best one.', 'In the toy model, width 2 finds an output of probability 0.38 where greedy finds 0.24.', 1, 8),
 'BLEU': ('A score for machine translation that counts word sequences shared with reference translations.', 'Reversing the source raised it from 25.9 to 30.6.', 1, 6),
 'Catalan number': ('Cₙ = (2n choose n)/(n + 1), the number of binary trees with n internal nodes.', 'C₄ = 14.', 2, 4),
 'causal mask': ('A mask that stops position i from attending to positions after i.', 'Row 1 of a 3 × 3 causal mask allows only position 1.', 3, 5),
 'counter (owed subtrees)': ('While reading prefix tokens, the number of subtrees still owed; it changes by arity − 1 per token.', 'add 2 mul 3 add 5 2: 1, 2, 1, 2, 1, 2, 1, 0.', 2, 3),
 'cross-entropy': ('The loss −Σ qₖ log pₖ between a target distribution q and a predicted one p.', '', 3, 11),
 'D(e, n)': ('The number of ways to complete a tree with e empty slots and n internal nodes still to place.', 'D(1, 2) = 6.', 2, 4),
 'decoder': ('The network that writes the output one token at a time, from the encoder’s summary and its own previous outputs.', '', 1, 5),
 'direct sum ⨁': ('The operator that joins the word lists of its two arguments; ⨂ distributes over it.', '(a⨁b)⨂c = a⨂c ⨁ b⨂c.', 4, 4),
 'embedding': ('A learned vector for each token of the vocabulary.', 'EmbeddingLayer[d, V] stores a V × d matrix.', 3, 10),
 'encoder': ('The network that reads the whole input and summarises it.', '', 1, 4),
 'epoch': ('One pass through the whole training set.', '', 1, 7),
 'Equation 2 (Lample and Charton)': ('A recurrence for the number of expressions with n internal nodes. As printed it has p₁ where p₁² is needed; the forms agree only when p₁ is 0 or 1.', 'L = 4, p₁ = 3, p₂ = 2, n = 3: true 19 756, printed 19 822.', 2, 4),
 'expand': ('Rewrite an expression as a ⨁ of words by distributing ⨂ over ⨁.', '(a⨁b)⨂(c⨁d) = ac ⨁ ad ⨁ bc ⨁ bd.', 4, 4),
 'feed-forward network (FFN)': ('The two-layer network applied to every position on its own: max(0, xW₁ + b₁)W₂ + b₂.', 'dmodel → 2048 → dmodel in the base transformer.', 3, 8),
 'forget gate': ('The LSTM gate f that decides how much of the old memory to keep.', 'After a pulse the memory decays like fᵗ.', 1, 3),
 'forward generation (FWD)': ('Making integration data by drawing f and asking a computer algebra system for its integral.', '', 2, 6),
 'gate (LSTM)': ('A vector of numbers between 0 and 1 that scales a signal: the input, forget and output gates of an LSTM.', '', 1, 3),
 'gradient clipping by norm': ('If the norm s of the whole gradient exceeds a threshold, rescale the gradient to that norm, keeping its direction.', 'g = (6, 8), threshold 5: (3, 4).', 1, 7),
 'grammar-constrained decoding': ('Decoding that only allows tokens which keep the output a valid prefix tree.', 'After ⨁ a b only </s> is allowed.', 4, 7),
 'greedy decoding': ('Decoding that always takes the most likely next token: a beam of width 1.', '', 1, 8),
 'hidden state': ('The vector h an LSTM shows to the next layer at each step.', '', 1, 3),
 'integration by parts (IBP)': ('∫ FG′ = FG − ∫ F′G; used to make new integration pairs from known ones.', '∫ x cos x dx = x sin x + cos x.', 2, 6),
 'key, query, value': ('In attention, each position asks a query, and every position offers a key (to be matched) and a value (to be averaged).', '', 3, 3),
 'label smoothing': ('Training against the target (1 − ε)·onehot + ε/K instead of a one-hot target.', 'K = 4, ε = 0.1: (0.025, 0.025, 0.925, 0.025).', 3, 11),
 'large Schröder number': ('Sₙ, the number of unary–binary trees with n internal nodes.', 'S₂ = 6, S₃ = 22.', 2, 4),
 'LayerNorm': ('Normalise each position’s vector to mean 0 and variance 1, then apply a learned scale and shift.', '(1, 2, 3) → (−1.2238, 0, 1.2238) with ε = 0.001.', 3, 8),
 'learning rate': ('The size of each gradient step.', '0.7 for 5 epochs, then halved every half epoch (Unit 1).', 1, 7),
 'length penalty': ('lp(Y) = ((5 + |Y|)/6)^α; dividing log-probabilities by it stops beam search from preferring short outputs.', 'α = 0.6: lp(2) = 1.097, lp(8) = 1.590.', 3, 13),
 'LSTM': ('Long short-term memory: a recurrent cell with a memory updated by addition and controlled by gates.', 'One step by hand: h′ = 0.1743, c′ = 0.2877.', 1, 3),
 'memory cell': ('The vector c an LSTM keeps for itself, updated as c′ = f c + i g.', '', 1, 3),
 'multi-head attention': ('h attentions in dmodel/h dimensions each, on different projections, joined and projected again.', 'Base model: 8 heads of 64; 1 050 624 parameters.', 3, 6),
 'NetSharedArray': ('A Wolfram array that several layers use as one; here, the shared embedding and output weights.', '', 3, 10),
 'oracle': ('An exact symbolic program that labels training data and checks the net’s answers.', 'equivalentQ in Unit 4; tnEquivalentQ in TensorNets.', 4, 4),
 'padding mask': ('A mask that hides padding at the end of short sequences.', '', 3, 5),
 'parameter': ('A number the training adjusts: a weight or a bias.', 'The paper of Unit 1 has 384 112 000.', 1, 9),
 'positional encoding': ('A code added to each embedding to give attention the word order; sines and cosines of the position.', 'PE(1) for dmodel = 4: (0.8415, 0.5403, 0.0100, 0.99995).', 3, 7),
 'prefix (Polish) notation': ('Writing each operator before its arguments, so no parentheses are needed.', '2 + 3 × (5 + 2) is add 2 mul 3 add 5 2.', 2, 3),
 'reversing the source': ('Feeding the input sentence backwards, which brings the first source and target words close together.', 'T = 3: distances {1, 3, 5} instead of {3, 3, 3}.', 1, 6),
 'scaled dot product': ('A query–key score divided by √dₖ, so its variance is 1 rather than dₖ.', '', 3, 4),
 'shared embeddings': ('One weight matrix for the input embedding, the output embedding and the projection before the softmax.', '', 3, 10),
 'softmax': ('pₖ = e^{zₖ}/Σⱼ e^{zⱼ}: turns scores into positive probabilities that sum to 1.', '(2, 1, 0) → (0.665, 0.245, 0.090).', 1, 5),
 'teacher forcing': ('During training the decoder reads the correct previous token, not its own guess.', '', 1, 5),
 'tensor product ⨂': ('The operator that multiplies every word of its left argument by every word of its right, keeping order; not commutative.', 'a⨂c and c⨂a are different words.', 4, 4),
 'thought vector': ('The final states of the encoder’s layers, which summarise the whole input.', '2 × 4 × 1000 = 8000 numbers in the paper of Unit 1.', 1, 4),
 'uniform sampling of trees': ('Drawing every tree with n internal nodes equally often, by choosing each node with probability proportional to D(e, n).', 'First choice for n = 2: unary 1/3, binary 2/3.', 2, 5),
 'verification': ('Checking an answer exactly instead of trusting the net: by differentiating (Unit 2) or by the oracle (Unit 4).', 'Beam 4 with checking: 97 % of test expressions solved in Unit 4.', 2, 7),
 'warm-up': ('A learning rate that rises linearly at the start of training, then falls.', 'Peak 6.99 × 10⁻⁴ at step 4000 for dmodel = 512.', 3, 12),
 'word (in ⨂/⨁ algebra)': ('An ordered ⨂-product of atoms; an expanded expression is a ⨁ of words.', 'a⨂b⨂d.', 4, 4),
}

def page(title, body, nav):
    return ('<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>%s — Computational Classnotes</title><style>%s%s</style></head><body>%s%s'
            '<footer style="margin-top:30px;font-size:9pt;color:#666">© 2026 Computational Classnotes · CC BY-NC-SA 4.0</footer></body></html>\n'
            % (html.escape(title), CSS, SCREEN, nav, body))

items = []
for term in sorted(TERMS, key=lambda t: t.lower().lstrip('⨂⨁ ')):
    d, ex, u, s = TERMS[term]
    link = 'units/%s.html#%s' % (U[u], {1: '', 2: ''}.get(0, ''))
    items.append('<dt id="%s">%s</dt><dd>%s%s <a href="units/%s.html">Unit %d §%d</a></dd>' % (
        html.escape(term.split(' (')[0].replace(' ', '-').lower()), html.escape(term), html.escape(d),
        (' <span class="ex">' + html.escape(ex) + '</span>') if ex else '', U[u], u, s))
nav = '<nav class="wtop"><a href="units/index.html">All units</a><a href="design.html">Design</a><a href="net-diagrams.html">Net diagrams</a><a href="../search.html">Search</a><a href="%s">Repository</a></nav>' % REPO
(ROOT / 'docs/glossary.html').write_text(page('Glossary · TensorNets units', '<h1>Glossary</h1><p>Every technical term of the four TensorNets units, defined in a sentence, with a small example where it helps, and a link to the section where it is worked out.</p><dl>%s</dl>' % ''.join(items), nav), encoding='utf-8')

UNITS = [
 (1, '01-seq2seq', 'unit1_seq2seq.nb', 'Reading a Sequence, Writing a Sequence', 'Deep LSTMs as encoder and decoder (Sutskever, Vinyals and Le, 2014): the LSTM cell by hand, the thought vector, reversing the source, the training recipe, beam search, the 384M parameter count.'),
 (2, '02-symbolic-data', 'unit2_symbolic_data.nb', 'Mathematics as Data', 'Lample and Charton (2019): prefix notation, counting and uniform sampling of trees, FWD/BWD/IBP integration data, verified answers, and a correction to their Equation 2.'),
 (3, '03-transformer', 'unit3_transformer.nb', 'Attention, Part by Part', 'Vaswani et al. (2017): every part of the transformer from plain arrays and from primitive Wolfram layers, checked against each other.'),
 (4, '04-build-your-own', 'unit4_build_your_own.nb', 'Building Our Own Net', 'A transformer assembled from Units 1–3 that expands ⨂/⨁ expressions, decoded with beam search and the prefix grammar, every answer checked by an exact oracle.'),
]
cards = ''.join('<div class="unit"><h3>Unit %d · %s</h3><p>%s</p><p><a href="%s.html">Read</a> · <a href="%s.pdf">PDF</a> · <a href="%s/blob/main/docs/units/%s.md">Markdown</a> · <a href="%s/blob/main/%s">Notebook %s</a></p></div>'
                % (n, html.escape(t), html.escape(d), s, s, REPO, s, REPO, nb, nb) for n, s, nb, t, d in UNITS)
nav2 = '<nav class="wtop"><a href="../glossary.html">Glossary</a><a href="../design.html">Design</a><a href="../net-diagrams.html">Net diagrams</a><a href="../../search.html">Search</a><a href="%s">Repository</a></nav>' % REPO
intro = ('<h1>TensorNets · Units of learning</h1><p>Four units on how the TensorNets nets are made. Each unit has a write-up (HTML, PDF, Markdown) and a Wolfram notebook with the same section numbers: '
         'a toolkit cell, then one section per idea, each with a worked example by hand and a check that the kernel returns. Units 1–3 follow one paper each; Unit 4 builds our own net from them.</p>'
         '<p>Open a notebook in Wolfram, choose Evaluation ▸ Quit Kernel ▸ Local, and evaluate it from top to bottom. The notebooks are saved with their outputs, so they can also be read without running.</p>')
(ROOT / 'docs/units/index.html').write_text(page('TensorNets · Units of learning', intro + cards, nav2), encoding='utf-8')
root_cards = cards.replace('href="0', 'href="docs/units/0')
(ROOT / 'index.html').write_text('<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">\n'
    '<title>TensorNets</title>\n<meta http-equiv="refresh" content="0; url=docs/units/index.html">\n</head><body style="font:16px system-ui;margin:40px">\n'
    '<p><a href="docs/units/index.html">TensorNets · Units of learning</a> · <a href="docs/design.html">design</a> · <a href="docs/glossary.html">glossary</a> · <a href="docs/net-diagrams.html">net diagrams</a> · <a href="search.html">search</a></p>\n'
    '<p>© 2026 Computational Classnotes · CC BY-NC-SA 4.0</p>\n</body></html>\n', encoding='utf-8')
# ---------- The Net Diagrams editor, shown live from ccn-nets ----------
nav3 = '<nav class="wtop"><a href="units/index.html">All units</a><a href="glossary.html">Glossary</a><a href="design.html">Design</a><a href="../search.html">Search</a><a href="%s">Repository</a></nav>' % REPO
editor_body = ('<h1>Net diagrams</h1>'
    '<p>Draw a small neural net and let the editor write the code that builds and trains it: a Wolfram notebook (our own code, and <code>NetTrain</code>) '
    'or a Python script (our own code, NumPy only). Its elements are feature groups, a hash, an input vector of slots, layers, dropout, joins and a softmax; '
    'it reads left to right, and wires are drawn from port to port.</p>'
    '<p>The editor belongs to the CCN agent free-form language. This page shows the live copy, so every change made there appears here at once. '
    'Besides small feed-forward nets such as SubjectNet, it draws transformers: tokens, embeddings, encoders, decoders and their outputs. '
    'Here it opens on tnParserNet, and keeps your changes on this page apart from the editor\u2019s other pages. Under <i>Examples</i> are owNet (Unit 4) and the other TensorNets nets, tnPolicyNet and tnValueNet, all drawn from their definitions; '
    'for each, the editor counts the weights as Wolfram does and writes the net as a Wolfram <code>NetGraph</code>.</p>'
    '<p><a href="%s">Open the editor in its own tab</a> · <a href="%s">How it works</a></p>'
    '<iframe src="%s" title="CCN Net Diagrams editor" style="width:100%%;height:820px;border:1px solid #c9d0db;border-radius:6px" loading="lazy"></iframe>'
    % (EDITOR, EDITOR_README, EDITOR))
(ROOT / 'docs/net-diagrams.html').write_text(page('Net diagrams · TensorNets', editor_body, nav3).replace('max-width:860px', 'max-width:1280px', 1), encoding='utf-8')

print(len(TERMS), 'terms')
