# STYLE — TensorNets units and documents

*How the write-ups, notebooks and site of this repository are made. Same house style as the Laplacian series.*

## 1. The write-up

- Title, date line, a lead paragraph, then **Scope and conventions** and **At a glance** (one figure).
- Numbered sections that match the notebook's sections; notebook §1 is the toolkit.
- Each idea: plain explanation, the formula, a **Worked by hand** paragraph with real numbers, then **Checked (notebook §n)** quoting what the kernel returned.
- Figures numbered in order, each with a caption that says what to see.
- Ends with exercises and answers, a summary, **Verified results** (a table) and **References** (items not opened marked †).
- Short sentences; define every term or link it to the glossary.

## 2. The companion notebook

- §1 Toolkit: one Input cell with every function. Then Section, Text, the formula, the worked example (highlighted) and short check cells.
- Every Input cell starts with its id comment: `(* id: <uuid> · <notebook> · cell n of m · updated <time> UTC *)`.
- Saved with outputs, so it can be read without running. Evaluated in a fresh `wolframscript` process; no messages allowed.

## 3. Files

- Markdown is the source; HTML (MathML) and PDF (A4, headless Chrome) are built by `tools/units/build_docs.py`.
- Notebook sources are `tools/units/unit*.py`, built by `tools/units/unitbuild.py`.
- Tokens such as `<s>`, `<e>` and `</s>` go in backticks in Markdown, or they vanish as HTML tags.

## 4. Site

- `index.html` redirects to the units index; `search.html` searches every write-up section, the design and the glossary.
- Rebuild after any change: `build_docs.py`, then `build_glossary.py`, then `build_search.py`.

## 5. Git

- Commits as Noor Hoshino <noor@compclassnotes.com>; no Claude attribution. Push only on "push".

---

© Computational Classnotes · CC BY-NC-SA 4.0
