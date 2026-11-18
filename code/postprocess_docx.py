"""Post-process pandoc docx: numbered display equations (3-column borderless table), numbered figure captions,
three-line tables. Usage: python3 postprocess_docx.py in.docx out.docx"""
import sys, copy, re
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
doc = Document(sys.argv[1])
body = doc.element.body


def borders(tbl, spec):
    tblPr = tbl._tbl.tblPr
    for old in tblPr.findall(qn("w:tblBorders")):
        tblPr.remove(old)
    b = OxmlElement("w:tblBorders")
    for e in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        el = OxmlElement(f"w:{e}")
        v = spec.get(e)
        el.set(qn("w:val"), "single" if v else "nil")
        if v:
            el.set(qn("w:sz"), str(v)); el.set(qn("w:color"), "000000")
        b.append(el)
    tblPr.append(b)


# 1) equations
eq = 0
for p in list(doc.paragraphs):
    omp = p._p.find(f".//{M}oMathPara")
    if omp is None:
        continue
    eq += 1
    t = doc.add_table(rows=1, cols=3)
    tblPr = t._tbl.tblPr
    for st in tblPr.findall(qn("w:tblStyle")):
        tblPr.remove(st)
    for lk in tblPr.findall(qn("w:tblLook")):
        tblPr.remove(lk)
    look = OxmlElement("w:tblLook"); look.set(qn("w:val"), "0000")
    for a in ["firstRow", "lastRow", "firstColumn", "lastColumn", "noHBand", "noVBand"]:
        look.set(qn("w:" + a), "1" if a.startswith("no") else "0")
    tblPr.append(look)
    borders(t, {})
    for c in t.rows[0].cells:
        tcPr = c._tc.get_or_add_tcPr(); bd = OxmlElement("w:tcBorders")
        for e in ["top", "left", "bottom", "right", "insideH", "insideV"]:
            el = OxmlElement(f"w:{e}"); el.set(qn("w:val"), "nil"); bd.append(el)
        tcPr.append(bd)
    widths = [Cm(1.5), Cm(13.0), Cm(1.5)]
    for i, w in enumerate(widths):
        t.columns[i].width = w; t.rows[0].cells[i].width = w
    mid = t.rows[0].cells[1].paragraphs[0]
    mid.alignment = WD_ALIGN_PARAGRAPH.CENTER
    mid._p.append(copy.deepcopy(omp))
    r = t.rows[0].cells[2].paragraphs[0]; r.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = r.add_run(f"({eq})"); run.font.size = Pt(11)
    for c in t.rows[0].cells:
        tcPr = c._tc.get_or_add_tcPr(); va = OxmlElement("w:vAlign"); va.set(qn("w:val"), "center"); tcPr.append(va)
    p._p.addprevious(t._tbl)
    p._p.getparent().remove(p._p)

# 2) figure captions
fig = 0
for p in doc.paragraphs:
    if p.style.name in ("Image Caption", "Caption") and p.text.strip():
        fig += 1
        first = p.runs[0] if p.runs else None
        new = OxmlElement("w:r"); rpr = OxmlElement("w:rPr"); bb = OxmlElement("w:b"); rpr.append(bb); new.append(rpr)
        tt = OxmlElement("w:t"); tt.set(qn("xml:space"), "preserve"); tt.text = f"Fig. {fig}. "; new.append(tt)
        p._p.insert(1 if p._p.pPr is not None else 0, new)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

# 3) data tables: three-line style, header bold, 9 pt
for t in doc.tables:
    if len(t.columns) == 3 and len(t.rows) == 1 and not t.rows[0].cells[0].text.strip():
        continue  # equation tables
    borders(t, {"top": 12, "bottom": 12})
    hdr = t.rows[0]
    for c in hdr.cells:
        tcPr = c._tc.get_or_add_tcPr(); bd = OxmlElement("w:tcBorders"); bt = OxmlElement("w:bottom")
        bt.set(qn("w:val"), "single"); bt.set(qn("w:sz"), "6"); bt.set(qn("w:color"), "000000"); bd.append(bt); tcPr.append(bd)
        for pp in c.paragraphs:
            for rr in pp.runs:
                rr.bold = True
    for row in t.rows:
        for c in row.cells:
            for pp in c.paragraphs:
                pp.paragraph_format.space_after = Pt(0); pp.paragraph_format.space_before = Pt(0)
                for rr in pp.runs:
                    rr.font.size = Pt(9)

# caption/table caption styling
for p in doc.paragraphs:
    if p.style.name == "Table Caption":
        p.paragraph_format.keep_with_next = True
        if p.runs and p.runs[0].text.startswith("Table"):
            m = re.match(r"(Table \d+\.)", p.runs[0].text)
doc.save(sys.argv[2])
print("equations", eq, "figures", fig, "tables", len(doc.tables))
