"""Build the site search index for the units, the design document and the glossary.

python3 tools/units/build_search.py   ->  search-index.json (search.html loads it)
Run after any write-up, glossary or design rebuild.
© Computational Classnotes · CC BY-NC-SA 4.0
"""
import re, json, html, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SITE = 'https://lossofgenerality.github.io/monoid-algebra-nets/'


def text(h):
    h = re.sub(r'<(script|style|svg|annotation)[^>]*>.*?</\1>', ' ', h, flags=re.S)
    h = re.sub(r'<(br|p|li|tr|div|h\d|dd|dt)[^>]*>', ' ', h)
    h = re.sub(r'<[^>]+>', '', h)
    return re.sub(r'\s+', ' ', html.unescape(h)).strip()


entries = []


def add(source, title, section, url, body):
    body = body.strip()
    if body:
        entries.append({'s': source, 't': title, 'h': section, 'u': url, 'x': body})


# write-ups (pandoc HTML): one entry per h2/h3 section
def writeup(path, url, source):
    s = path.read_text(encoding='utf-8')
    title = text(re.search(r'<h1[^>]*>(.*?)</h1>', s, re.S).group(1))
    body = s[s.index('</h1>'):]
    parts = re.split(r'(<h[23][^>]*id="[^"]*"[^>]*>.*?</h[23]>)', body, flags=re.S)
    add(source, title, 'Introduction', url, text(parts[0])[:4000])
    for i in range(1, len(parts), 2):
        hid = re.search(r'id="([^"]*)"', parts[i]).group(1)
        add(source, title, text(parts[i]), url + '#' + hid, text(parts[i + 1])[:6000])


for p in sorted((ROOT / 'docs/units').glob('[0-9][0-9]-*.html')):
    writeup(p, SITE + 'docs/units/' + p.name, 'Unit')
writeup(ROOT / 'docs/design.html', SITE + 'docs/design.html', 'Design')

# glossary: one entry per term
g = (ROOT / 'docs/glossary.html').read_text(encoding='utf-8')
for m in re.finditer(r'<section class="g" id="([^"]+)"><h2>(.*?)</h2><p>(.*?)</p>', g, re.S):
    add('Glossary', 'Glossary', text(m.group(2)), SITE + 'docs/glossary.html#' + m.group(1), text(m.group(3)))

(ROOT / 'search-index.json').write_text(json.dumps(entries, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
print(len(entries), 'entries;', sum(len(e['x']) for e in entries) // 1000, 'k characters')
