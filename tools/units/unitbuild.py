"""python3 unitbuild.py unit1 [unit2 ...] -> /mnt/user-data/outputs/<FILE>.nb
A unit module defines FILE, TITLE, SUBTITLE and CELLS = [(style, text), ...].
Styles: Title, Subtitle, Section, Subsection, Text, Formula, Input, Figure (caption for the figure above).
Every Input cell gets the house identifier comment; figures are numbered in order."""
import sys, importlib, json, os, re, uuid, datetime

def wl_escape(s):
    out = []
    for ch in s:
        o = ord(ch)
        if ch == '\\': out.append('\\\\')
        elif ch == '"': out.append('\\"')
        elif ch == '\n': out.append('\\n')
        elif o > 0xFFFF: out.append('\\|%06x' % o)
        elif o > 127: out.append('\\:%04x' % o)
        else: out.append(ch)
    return ''.join(out)

_SUB = re.compile(r'([A-Za-zͰ-Ͽ]+)_(\{[^}]*\}|[A-Za-z0-9Ͱ-Ͽ]+)')
def text_cell(text, style):
    parts, i = [], 0
    for m in _SUB.finditer(text):
        parts.append(text[i:m.start()]); sb = m.group(2)
        parts.append((m.group(1), sb[1:-1] if sb.startswith('{') else sb)); i = m.end()
    parts.append(text[i:]); parts = [p for p in parts if p != '']
    if all(isinstance(p, str) for p in parts):
        return 'Cell["%s", "%s"]' % (wl_escape(text), style)
    items = ['"%s"' % wl_escape(p) if isinstance(p, str) else
             'Cell[BoxData[SubscriptBox["%s", "%s"]]]' % (wl_escape(p[0]), wl_escape(p[1])) for p in parts]
    return 'Cell[TextData[{%s}], "%s"]' % (', '.join(items), style)

ORANGE = 'RGBColor[0.83, 0.34, 0.11]'
def build(mod):
    m = importlib.import_module(mod)
    ids = json.load(open('ids.json')) if os.path.exists('ids.json') else {}
    now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M')
    n_in = sum(1 for s, _ in m.CELLS if s == 'Input')
    cells, k, fig = [], 0, 0
    for s, t in m.CELLS:
        if s == 'Input':
            k += 1; key = '%s/%d' % (m.FILE, k)
            ids.setdefault(key, str(uuid.uuid4()))
            head = '(* id: %s · %s · cell %d of %d · updated %s UTC *)\n' % (ids[key], m.FILE, k, n_in, now)
            cells.append('Cell["%s", "Input"]' % wl_escape(head + t))
        elif s == 'Figure':
            fig += 1
            cells.append('Cell[TextData[{StyleBox["Fig. %d. ", FontWeight -> "Bold", FontColor -> %s], "%s"}], "Text"]'
                         % (fig, ORANGE, wl_escape(t)))
        elif s == 'Formula':
            cells.append('Cell["%s", "Text", TextAlignment -> Center, FontFamily -> "Times", FontSize -> 15, FontSlant -> "Italic"]' % wl_escape(t))
        elif s == 'Hand':
            cells.append(text_cell(t, 'Text').replace('"Text"]', '"Text", Background -> RGBColor[1, 0.97, 0.92], CellFrame -> {{3, 0}, {0, 0}}, CellFrameColor -> %s]' % ORANGE))
        else:
            cells.append(text_cell(t, s))
    json.dump(ids, open('ids.json', 'w'), indent=0)
    nb = ('(* Content-type: application/vnd.wolfram.mathematica *)\n(* %s - Computational Classnotes - CC BY-NC-SA 4.0 *)\n\n'
          'Notebook[{\n%s\n},\nWindowTitle -> "%s",\nStyleDefinitions -> "Default.nb"\n]\n'
          % (m.TITLE.encode('ascii', 'replace').decode(), ',\n'.join(cells), wl_escape(m.TITLE)))
    open(os.path.join(os.environ.get('UNIT_OUT', '.'), '%s.nb' % m.FILE), 'w', encoding='ascii').write(nb)
    print(m.FILE, 'cells', len(cells), 'inputs', n_in, 'figures', fig)

if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    for mod in sys.argv[1:]: build(mod)
