# CONTEXT — TensorNets

*Updated 24 November 2026 · What this repository is, where everything lives, and what is still open.*

By Dara O Shayda and Hikari Noor Hoshino (星野ヌール)  
AI System Programming Group

---

## 1. What the project is

Neural nets that **parse, evaluate and simplify** tensor-algebra expressions in ⨂ (tensor product) and ⨁ (direct sum), such as `(a⨁b)⨂c`. There are three nets, each paired with an exact symbolic oracle that labels its data and checks its answers: `tnParserNet` (encoder–decoder, tokens → prefix tree), `tnPolicyNet` (which rewrite rule, at which node) and `tnValueNet` (dimension, term count, steps). Wolfram Language first; Python later. The design is in `DESIGN-tensornets.md`.

Four **units of learning** explain how such nets are made, following one paper each (Sutskever 2014, Lample & Charton 2019, Vaswani 2017) and then building our own.

## 2. Deliverables and where they are

### Files

| Path | Holds |
|---|---|
| `ff_tensornets_new.nb` | The current code, every function in one Input cell |
| `ff_tensornets_test.nb`, `ff_tensornets_new_testlog.nb` | The tests (22) and their latest results |
| `ff_tensornets_smallnets.nb` | Small development nets |
| `ff_tensornets.nb`, `ff_tensornets_testlog.nb` | The earlier version, kept unchanged |
| `ref1_…`, `ref2_…`, `ref3_…` `.nb` | The three papers' models, with small demo nets (`_nets.nb`) and logs (`_log.nb`) |
| `unit1_seq2seq.nb` … `unit4_build_your_own.nb` | The units' notebooks, saved with outputs; `unit4_build_your_own_nets.nb` holds the trained net |
| `lesson*.nb` | The earlier single-cell lessons, superseded by the units, kept until their removal is approved |
| `DESIGN-tensornets.md`, `.pdf` | The design |
| `docs/units/` | The unit write-ups (`.md`, `.html`, `.pdf`), their figures (`img/`) and the units index |
| `docs/glossary.html`, `docs/design.html` | Glossary of 55 terms; the design as a site page |
| `docs/net-diagrams.html` | The CCN Net Diagrams editor, shown live from `ccn-agent-free-form-language` (`editor/net-diagram-editor.html` on its GitHub Pages site); built by `build_glossary.py`. Nothing is copied: a change to the editor there appears here at once |
| `index.html`, `search.html`, `search-index.json` | The site: landing (redirects to the units), search page and its index |
| `brand/` | CCN design tokens and fonts used by the search page |
| `tools/units/` | Builders: `unitbuild.py` + `unit1.py`…`unit4.py` (notebooks), `build_docs.py` (HTML, PDF), `build_glossary.py` (glossary, units index, landing), `build_search.py` (search index) |
| `python/` | `tensor_parser.py` and its `CONTEXT.md` |

The reference papers themselves stay out of git; they are kept only in the local TensorNets folder.

### Site

GitHub Pages from `main`, root folder: https://lossofgenerality.github.io/tensor-algebra-nets/. Rebuild order after a change: `build_docs.py` → `build_glossary.py` → `build_search.py`.

The Net diagrams page depends on the editor's address, https://lossofgenerality.github.io/ccn-agent-free-form-language/editor/net-diagram-editor.html, which is set in `build_glossary.py` (`EDITOR`). If the editor ever moves, change it there and rebuild.

## 3. Unit structure

Each unit has a write-up and a notebook with the same section numbers. The notebook starts with **§1 Toolkit**, one Input cell with every function, then one section per idea: an explanation, the formula, a highlighted *worked by hand* example and a short check cell. Each unit ends with exercises (answers computed), a summary, a table of verified results and references.

## 4. Conventions

- Work only in `~/Desktop/Research/TensorNets` on the Mac; notebooks only, no `.wl`.
- Nothing printed to the chat; results go into notebooks.
- No writes to the Wolfram Cloud without approval; no deletions without authorisation; never `DeleteObject`.
- Commits as Noor Hoshino <noor@compclassnotes.com>, with no Claude attribution; push only on the owner's "push".
- © Computational Classnotes · CC BY-NC-SA 4.0 on every piece of work.

## 5. Technical lessons

- `NotebookEvaluate` under `wolframscript` writes text forms (`-Graphics-`); the units are evaluated cell by cell with `ToExpression` on the source string and `ToBoxes` for the outputs.
- A cell's id comment followed by code must be evaluated from its source string; boxes joined by newlines evaluate as a product.
- `NetInitialize` sets LSTM forget-gate biases to 1 whatever the method; the paper's uniform start needs them reset.
- `LongShortTermMemoryLayer` states are reached through `NetGraph` ports; `AttentionLayer` takes `"ScoreRescaling" -> "DimensionSqrt"`.
- Equation 2 of Lample and Charton needs p₁² in place of p₁ when p₁ > 1.

## 6. Open issues

- Full-size training (`tnTrainAll[]`) not yet run.
- The Cloud release has not been replaced with `ff_tensornets_new.nb`; empty objects in `AI/freeform/ff/tensornets/nets/` and an empty `Test/` folder await the owner's decision.
- The 23 references of the design document have not been checked one by one.

---

© Computational Classnotes · CC BY-NC-SA 4.0
