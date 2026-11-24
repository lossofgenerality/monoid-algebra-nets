"""python3 tools/units/build_docs.py docs/units/01-seq2seq.md [...]
Writes <name>.html (MathML) and <name>.pdf (A4, headless Chrome) next to each markdown file.
© Computational Classnotes · CC BY-NC-SA 4.0"""
import sys, subprocess, re, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
CSS = (ROOT / 'tools/units/pdf-style.css').read_text(encoding='utf-8')
SCREEN = '''
body{max-width:860px;margin:0 auto;padding:28px 22px 60px;background:#fff}
img{max-width:100%;height:auto}
nav.wtop{font-size:10pt;margin-bottom:14px;display:flex;gap:14px;flex-wrap:wrap}
nav.wtop a{color:rgb(212,86,28)}
@media print{nav.wtop{display:none} footer{display:none}}
'''
REPO = 'https://github.com/lossofgenerality/tensor-algebra-nets/blob/main/'
NB = {'01-seq2seq': 'unit1_seq2seq.nb', '02-symbolic-data': 'unit2_symbolic_data.nb',
      '03-transformer': 'unit3_transformer.nb', '04-build-your-own': 'unit4_build_your_own.nb'}
for md in sys.argv[1:]:
    p = pathlib.Path(md).resolve()
    body = subprocess.run(['pandoc', str(p), '--mathml', '-t', 'html5'], capture_output=True, text=True, check=True).stdout
    title = re.search(r'^# (.+)$', p.read_text(encoding='utf-8'), re.M).group(1)
    nb = NB.get(p.stem, '')
    nav = ('<nav class="wtop"><a href="index.html">All units</a><a href="%s">PDF</a><a href="%s">Markdown</a>%s'
           '<a href="../glossary.html">Glossary</a><a href="../../search.html">Search</a><a href="%s">Repository</a></nav>'
           % (p.with_suffix('.pdf').name, REPO + str(p.relative_to(ROOT)),
              ('<a href="%s">Notebook %s</a>' % (REPO + nb, nb)) if nb else '', REPO.replace('/blob/main/', '')))
    html = ('<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>%s — Computational Classnotes</title><style>%s%s</style></head><body>%s%s'
            '<footer style="margin-top:30px;font-size:9pt;color:#666">© 2026 Computational Classnotes · CC BY-NC-SA 4.0</footer></body></html>\n'
            % (title, CSS, SCREEN, nav, body))
    out = p.with_suffix('.html'); out.write_text(html, encoding='utf-8')
    pdf = p.with_suffix('.pdf')
    subprocess.run(['google-chrome', '--headless=new', '--no-sandbox', '--disable-gpu', '--no-pdf-header-footer',
                    '--print-to-pdf=%s' % pdf, out.as_uri()], capture_output=True, check=True, timeout=180)
    print(out.name, pdf.name)
