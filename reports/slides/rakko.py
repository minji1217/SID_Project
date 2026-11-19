"""랏코(Rakko) 16:9 슬라이드 공통 헬퍼 (python-pptx).

규칙은 reports/slides/DESIGN.md. 좌표·크기는 모두 1920×1080 캔버스의 px로 받는다.
- 글자: Pretendard 굵기별 폰트 이름 (Pretendard, Pretendard Medium, … Bold)
- 편집 설정: 텍스트 자동 맞춤 끔(noAutofit), 개체 이름, 저장 후 snap 끔
- 수식: pdflatex + pdftocairo 로 굵은 LaTeX 투명 PNG
"""
from __future__ import annotations

import copy
import hashlib
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from lxml import etree
from PIL import Image, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu

# ── 토큰 (DESIGN.md 앞부분과 같은 값) ─────────────────────────────
C = {
    "primary": "14181F", "on-primary": "FFFFFF", "ink": "14181F", "ink-strong": "000000",
    "charcoal": "2E3440", "slate": "5B6577", "steel": "5B6577", "stone": "8A93A3", "muted": "8A93A3",
    "canvas": "FFFFFF", "surface": "F4F6F9", "surface-soft": "F8F9FB",
    "hairline": "D5DAE1", "hairline-soft": "E3E6EB",
    "brand-blue": "4C7BF4", "brand-blue-deep": "1F3F99", "brand-blue-700": "2F55C8",
    "brand-blue-200": "DCE6FD", "brand-cyan": "A9C8FB", "brand-purple": "6E56CF",
    "brand-lavender": "ECE8FA", "amber": "F2B33D", "error": "D45656",
    "success-bg": "E3F5EA", "success-text": "1E7A46",
}
# Windows·LibreOffice가 찾는 이름(legacy family). Bold는 "Pretendard" + b="1"
FACE = {400: "Pretendard", 500: "Pretendard Medium", 600: "Pretendard SemiBold",
        700: "Pretendard", 800: "Pretendard ExtraBold", 900: "Pretendard Black"}
FONT_FILE = {400: "Regular", 500: "Medium", 600: "SemiBold", 700: "Bold", 800: "ExtraBold", 900: "Black"}
FONT_DIR = Path.home() / ".fonts" / "pretendard"
T = {  # typography: size, weight, line height, letter spacing(em)
    "cover-title": (112, 800, 1.10, -0.03), "section-title": (96, 800, 1.10, -0.03),
    "stat-giant": (160, 900, 1.0, -0.03), "stat-lg": (72, 800, 1.05, -0.02),
    "title": (56, 700, 1.20, -0.02), "subtitle": (28, 500, 1.40, -0.01), "chapter": (22, 600, 1.30, 0),
    "heading-lg": (40, 700, 1.25, -0.02), "heading-md": (32, 700, 1.30, -0.01),
    "card-title": (26, 600, 1.35, 0), "body-lg": (28, 400, 1.5, 0), "body": (24, 400, 1.5, 0),
    "body-strong": (24, 700, 1.5, 0), "table": (22, 400, 1.45, 0), "caption": (20, 400, 1.5, 0),
    "footer": (18, 400, 1.4, 0), "badge": (18, 600, 1.3, 0),
}
# 사용자 선호: 모든 글자를 한 단계 두껍게 (2026-11-18). 토큰 굵기 → 실제 굵기
WEIGHT_UP = {400: 500, 500: 600, 600: 700, 700: 800, 800: 900, 900: 900}
W, H = 1920, 1080
EMU_PX = 12192000 // W  # 6350
MARGIN = 96


def E(px: float) -> Emu:
    return Emu(int(round(px * EMU_PX)))


def rgb(tok: str) -> RGBColor:
    return RGBColor.from_string(C.get(tok, tok))


# ── 글자 폭 재기 (한 줄 넘침 확인용) ─────────────────────────────
_font_cache: dict = {}


def text_width(text: str, size: float, weight: int = 400) -> float:
    weight = WEIGHT_UP.get(weight, weight)
    key = (size, weight)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(str(FONT_DIR / f"Pretendard-{FONT_FILE[weight]}.otf"), int(round(size * 4)))
    return _font_cache[key].getlength(text) / 4


WARNINGS: list[str] = []


# ── 런·문단 ──────────────────────────────────────────────────────
_MARK = re.compile(r"(\*\*.+?\*\*|\[\[.+?\]\])")


def _set_font(run, weight: int):
    weight = WEIGHT_UP.get(weight, weight)
    rPr = run._r.get_or_add_rPr()
    rPr.set("b", "1" if weight == 700 else "0")
    rPr.set("altLang", "en-US")
    # 기준 덱처럼 pitchFamily·charset 를 같이 적는다 (Windows PowerPoint 글꼴 찾기). ea 는 한글(129 → -127)
    for tag, cs_ in (("a:latin", "0"), ("a:ea", "-127"), ("a:cs", "0")):
        el = rPr.find(qn(tag))
        if el is None:
            el = etree.SubElement(rPr, qn(tag))
        el.set("typeface", FACE[weight])
        el.set("pitchFamily", "34")
        el.set("charset", cs_)


def add_runs(p, text: str, size: float, weight: int, color: str, ls_em: float = 0):
    """**굵게(ink)** · [[굵게 파랑]] 표시를 런으로 나눈다."""
    for part in _MARK.split(text):
        if not part:
            continue
        w, col = weight, color
        if part.startswith("**"):
            part, w, col = part[2:-2], max(weight, 700), "ink"
        elif part.startswith("[["):
            part, w, col = part[2:-2], max(weight, 700), "brand-blue-deep"
        r = p.add_run()
        r.text = part
        f = r.font
        r._r.get_or_add_rPr().set("sz", str(int(round(size / 2 * 100))))
        f.color.rgb = rgb(col)
        if ls_em:
            r._r.get_or_add_rPr().set("spc", str(int(round(ls_em * size / 2 * 100))))
        r._r.get_or_add_rPr().set("lang", "ko-KR")
        _set_font(r, w)


def _plain(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*|\[\[(.+?)\]\]", lambda m: m.group(1) or m.group(2), text)


def _no_autofit(tf, l=0, t=0, r=0, b=0, anchor="t", wrap=True):
    bp = tf._txBody.find(qn("a:bodyPr"))
    for k, v in (("lIns", l), ("tIns", t), ("rIns", r), ("bIns", b)):
        bp.set(k, str(int(round(v * EMU_PX))))
    bp.set("wrap", "square" if wrap else "none")
    bp.set("anchor", {"t": "t", "m": "ctr", "b": "b"}[anchor])
    for child in list(bp):
        if child.tag in (qn("a:spAutoFit"), qn("a:normAutofit"), qn("a:noAutofit")):
            bp.remove(child)
    etree.SubElement(bp, qn("a:noAutofit"))


def _bullet(p, color="brand-blue-deep", indent_px=26, char="•"):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(int(indent_px * EMU_PX)))
    pPr.set("indent", str(int(-indent_px * EMU_PX)))
    for tag in ("a:buClr", "a:buFont", "a:buChar", "a:buNone"):
        el = pPr.find(qn(tag))
        if el is not None:
            pPr.remove(el)
    bc = etree.SubElement(pPr, qn("a:buClr"))
    etree.SubElement(bc, qn("a:srgbClr")).set("val", C[color])
    bf = etree.SubElement(pPr, qn("a:buFont"))
    bf.set("typeface", "Pretendard")
    etree.SubElement(pPr, qn("a:buChar")).set("char", char)


def text(slide, name, x, y, w, h, paras, style="body", color="charcoal", align="l", anchor="t",
         bullets=False, gap=0, size=None, weight=None, lh=None, pad=(0, 0, 0, 0), one_line=False):
    """paras: 문자열 또는 문자열 목록(문단). 문단마다 스타일을 바꾸려면 (text, dict) 튜플."""
    sz0, wt0, lh0, ls0 = T[style]
    size, weight, lh = size or sz0, weight or wt0, lh or lh0
    tb = slide.shapes.add_textbox(E(x), E(y), E(w), E(h))
    tb.name = name
    tf = tb.text_frame
    _no_autofit(tf, *pad, anchor=anchor)
    if isinstance(paras, str):
        paras = [paras]
    for i, item in enumerate(paras):
        opt = {}
        if isinstance(item, tuple):
            item, opt = item
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[opt.get("align", align)]
        p.line_spacing = opt.get("lh", lh)
        if i > 0 and opt.get("gap", gap):
            p.space_before = E(opt.get("gap", gap))
        psize = opt.get("size", size)
        add_runs(p, item, psize, opt.get("weight", weight), opt.get("color", color), opt.get("ls", ls0))
        if opt.get("bullet", bullets):
            _bullet(p, indent_px=opt.get("indent", 26))
        # 넘침 확인
        avail = w - pad[0] - pad[2] - (26 if opt.get("bullet", bullets) else 0)
        tw = text_width(_plain(item), psize, opt.get("weight", weight))
        if (one_line or opt.get("one_line")) and tw > avail:
            WARNINGS.append(f"[한 줄 넘침] {name}: '{_plain(item)[:30]}' {tw:.0f}px > {avail:.0f}px")
    return tb


# ── 도형 ────────────────────────────────────────────────────────
def rect(slide, name, x, y, w, h, fill=None, line=None, radius=0, line_w=1, dash=None, shape=None):
    st = shape or (MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE)
    s = slide.shapes.add_shape(st, E(x), E(y), E(w), E(h))
    s.name = name
    if radius and st == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = min(0.5, radius / min(w, h))
    if fill:
        s.fill.solid()
        s.fill.fore_color.rgb = rgb(fill)
    else:
        s.fill.background()
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = E(line_w)
        if dash:
            ln = s._element.spPr.find(qn("a:ln"))
            etree.SubElement(ln, qn("a:prstDash")).set("val", dash)
    else:
        s.line.fill.background()
    _drop_style(s)
    tf = s.text_frame
    _no_autofit(tf, 12, 8, 12, 8, anchor="m")
    return s


def _drop_style(s):
    st = s._element.find(qn("p:style"))
    if st is not None:
        s._element.remove(st)
    spPr = s._element.spPr
    if spPr.find(qn("a:effectLst")) is None:
        etree.SubElement(spPr, qn("a:effectLst"))


def box(slide, name, x, y, w, h, paras, fill=None, line=None, radius=16, style="body", color="charcoal",
        align="c", anchor="m", pad=(16, 10, 16, 10), line_w=1, gap=0, shape=None, dash=None, wrap=True):
    """글자가 들어간 도형 (노드·칩·카드)."""
    s = rect(slide, name, x, y, w, h, fill, line, radius, line_w, dash=dash, shape=shape)
    tf = s.text_frame
    _no_autofit(tf, *pad, anchor=anchor, wrap=wrap)
    sz, wt, lh, ls = T[style]
    if isinstance(paras, str):
        paras = [paras]
    for i, item in enumerate(paras):
        opt = {}
        if isinstance(item, tuple):
            item, opt = item
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[opt.get("align", align)]
        p.line_spacing = opt.get("lh", lh)
        if i > 0 and opt.get("gap", gap):
            p.space_before = E(opt.get("gap", gap))
        psize, pw = opt.get("size", sz), opt.get("weight", wt)
        add_runs(p, item, psize, pw, opt.get("color", color), opt.get("ls", ls))
        tw = text_width(_plain(item), psize, pw)
        if tw > w - pad[0] - pad[2] and opt.get("one_line", True) and "\n" not in item:
            WARNINGS.append(f"[상자 넘침] {name}: '{_plain(item)[:30]}' {tw:.0f}px > {w - pad[0] - pad[2]:.0f}px")
    return s


def pill(slide, name, x, y, text_, fill="primary", color="on-primary", h=32, padx=14, style="badge"):
    sz, wt, _, _ = T[style]
    w = text_width(text_, sz, wt) + 2 * padx + 6
    return box(slide, name, x, y, w, h, text_, fill=fill, radius=h / 2, style=style, color=color,
               pad=(2, 0, 2, 0), wrap=False), w


def arrow(slide, name, pts, color="stone", width_pt=3.0, head=True, dash=None, head_size="med"):
    """pts: [(x,y), ...] px. 꺾은선 화살표 (머리는 선의 일부)."""
    fb = slide.shapes.build_freeform(E(pts[0][0]), E(pts[0][1]), scale=1.0)
    fb.add_line_segments([(E(px), E(py)) for px, py in pts[1:]], close=False)
    s = fb.convert_to_shape()
    s.name = name
    s.fill.background()
    s.line.color.rgb = rgb(color)
    s.line.width = Emu(int(width_pt * 12700))
    _drop_style(s)
    _no_autofit(s.text_frame)
    ln = s._element.spPr.find(qn("a:ln"))
    if dash:
        etree.SubElement(ln, qn("a:prstDash")).set("val", dash)
    if head:
        te = etree.SubElement(ln, qn("a:tailEnd"))
        te.set("type", "triangle"); te.set("w", head_size); te.set("len", head_size)
    return s


def hline(slide, name, x, y, w, color="hairline-soft", width_px=1):
    s = rect(slide, name, x, y, w, max(width_px, 1), fill=color)
    return s


def dot(slide, name, cx, cy, r, fill="canvas", line=None, line_w=2):
    return rect(slide, name, cx - r, cy - r, 2 * r, 2 * r, fill, line, line_w=line_w, shape=MSO_SHAPE.OVAL)


# ── 표 ──────────────────────────────────────────────────────────
def _cell_border(tc, sides: dict):
    tcPr = tc.get_or_add_tcPr()
    for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        old = tcPr.find(qn(tag))
        if old is not None:
            tcPr.remove(old)
    insert_at = 0
    for tag, side in (("a:lnL", "L"), ("a:lnR", "R"), ("a:lnT", "T"), ("a:lnB", "B")):
        ln = etree.Element(qn(tag))
        col = sides.get(side)
        if col:
            ln.set("w", str(EMU_PX)); ln.set("cap", "flat"); ln.set("cmpd", "sng"); ln.set("algn", "ctr")
            sf = etree.SubElement(ln, qn("a:solidFill"))
            etree.SubElement(sf, qn("a:srgbClr")).set("val", C[col])
        else:
            ln.set("w", "0")
            etree.SubElement(ln, qn("a:noFill"))
        tcPr.insert(insert_at, ln)
        insert_at += 1


def table(slide, name, x, y, col_w, row_h, rows, header=True, highlight_rows=(), align=None,
          style="table", head_style="table", bold_cols=(), cell_colors=None):
    """rows: [[셀 문자열]]. 첫 행이 머리글. cell_colors: {(r,c): fill 토큰}."""
    nr, nc = len(rows), len(rows[0])
    gs = slide.shapes.add_table(nr, nc, E(x), E(y), E(sum(col_w)), E(row_h * nr))
    gs.name = name
    tbl = gs.table
    tblPr = tbl._tbl.tblPr
    for k in ("firstRow", "bandRow"):
        tblPr.set(k, "0")
    sid = tblPr.find(qn("a:tableStyleId"))
    if sid is not None:
        sid.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"  # No Style, No Grid
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = E(cw)
    for i in range(nr):
        tbl.rows[i].height = E(row_h)
        for j in range(nc):
            cell = tbl.cell(i, j)
            is_head = header and i == 0
            fill = "surface" if is_head else ("brand-blue-200" if i in highlight_rows else None)
            if cell_colors and (i, j) in cell_colors:
                fill = cell_colors[(i, j)]
            if fill:
                cell.fill.solid(); cell.fill.fore_color.rgb = rgb(fill)
            else:
                cell.fill.background()
            cell.margin_left = cell.margin_right = E(14)
            cell.margin_top = cell.margin_bottom = E(4)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            _bp = tf._txBody.find(qn("a:bodyPr"))
            if _bp.find(qn("a:noAutofit")) is None:
                etree.SubElement(_bp, qn("a:noAutofit"))
            p = tf.paragraphs[0]
            a = (align[j] if align else ("l" if j == 0 else "c"))
            p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[a]
            sz, wt, lh, _ = T[head_style if is_head else style]
            if is_head:
                wt, col = 600, "steel"
            else:
                col = "charcoal"
                if j in bold_cols:
                    wt, col = 700, "ink"
            p.line_spacing = 1.2
            for k, line_ in enumerate(str(rows[i][j]).split("\n")):
                pp = p if k == 0 else tf.add_paragraph()
                pp.alignment = p.alignment
                pp.line_spacing = 1.2
                add_runs(pp, line_, sz, wt, col)
            _cell_border(cell._tc, {"B": "hairline-soft"} if i < nr - 1 else {"B": "hairline"})
            if i == 0:
                _cell_border(cell._tc, {"T": "hairline", "B": "hairline"})
    return gs


# ── 수식 (굵은 LaTeX 이미지) ───────────────────────────────────
FORMULA_DIR = Path(tempfile.gettempdir()) / "rakko_formulas"


def formula_png(tex: str, color: str = "ink") -> Path:
    FORMULA_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5((tex + color).encode()).hexdigest()[:12]
    out = FORMULA_DIR / f"f_{key}.png"
    if out.exists():
        return out
    doc = (r"\documentclass[12pt,border=2pt]{standalone}" "\n"
           r"\usepackage{amsmath,amssymb,bm,xcolor}" "\n"
           rf"\definecolor{{fc}}{{HTML}}{{{C.get(color, color)}}}"
           rf"\definecolor{{hl}}{{HTML}}{{{C['brand-purple']}}}"
           rf"\definecolor{{bl}}{{HTML}}{{{C['brand-blue-deep']}}}" "\n"
           r"\begin{document}\color{fc}{\boldmath $\displaystyle " + tex + r" $}\end{document}")
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "f.tex").write_text(doc)
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "f.tex"], cwd=d, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"LaTeX 실패: {tex}\n{r.stdout[-800:]}")
        subprocess.run(["pdftocairo", "-png", "-transp", "-r", "576", "-singlefile", "f.pdf", "f"], cwd=d, check=True)
        shutil.copy(Path(d) / "f.png", out)
    return out


def formula(slide, name, x, y, tex, alt, scale=1.1, color="ink", anchor="tl", max_w=None):
    """12pt@576dpi PNG: 1 PNG px = 0.25 slide px (24px 글자) × scale. anchor: tl | ml(왼쪽 가운데) | mc"""
    png = formula_png(tex, color)
    pw, ph = Image.open(png).size
    w, h = pw * 0.25 * scale, ph * 0.25 * scale
    if max_w and w > max_w:
        h, w = h * max_w / w, max_w
    if anchor == "ml":
        y = y - h / 2
    elif anchor == "mc":
        x, y = x - w / 2, y - h / 2
    pic = slide.shapes.add_picture(str(png), E(x), E(y), E(w), E(h))
    pic.name = name
    pic._element.nvPicPr.cNvPr.set("descr", alt)
    return pic, w, h


# ── 슬라이드 틀 ─────────────────────────────────────────────────
class Deck:
    def __init__(self, total: int, footer_left: str):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = E(W), E(H)
        self.blank = self.prs.slide_layouts[6]
        self.total = total
        self.footer_left = footer_left
        self.n = 0

    def new(self, notes: str = ""):
        s = self.prs.slides.add_slide(self.blank)
        self.n += 1
        if notes:
            s.notes_slide.notes_text_frame.text = notes
        return s

    def footer(self, s, left=None, page=True):
        hline(s, "Footer rule", MARGIN, 984, W - 2 * MARGIN)
        text(s, "Footer left", MARGIN, 996, 1400, 28, left or self.footer_left, "footer", "muted", one_line=True)
        if page:
            text(s, "Page number", W - MARGIN - 200, 996, 200, 28, f"{self.n:02d} / {self.total:02d}", "footer", "muted", align="r")

    def frame(self, s, chap_no: str, chap: str, title: str, subtitle: str, footer_left=None):
        pill(s, "Chapter number", MARGIN, 52, chap_no)
        text(s, "Chapter name", MARGIN + 64, 48, 900, 40, chap, "chapter", "steel", anchor="m")
        text(s, "Title", MARGIN, 112, W - 2 * MARGIN, 72, title, "title", "ink", anchor="m", one_line=True)
        text(s, "Subtitle", MARGIN, 192, W - 2 * MARGIN, 40, subtitle, "subtitle", "steel", anchor="m", one_line=True)
        hline(s, "Header rule", MARGIN, 256, W - 2 * MARGIN)
        self.footer(s, footer_left)

    def save(self, path: Path):
        self.prs.save(str(path))
        _postprocess(path)


def _theme_fonts(xml: str) -> str:
    """테마 major/minor 글꼴을 Pretendard 로 (latin·ea·한글 script)."""
    xml = re.sub(r'(<a:(?:majorFont|minorFont)>\s*)<a:latin typeface="[^"]*"[^>]*/>\s*<a:ea typeface="[^"]*"[^>]*/>',
                 r'\1<a:latin typeface="Pretendard" pitchFamily="34" charset="0"/><a:ea typeface="Pretendard" pitchFamily="34" charset="-127"/>',
                 xml)
    xml = re.sub(r'<a:font script="Hang" typeface="[^"]*"/>', '<a:font script="Hang" typeface="Pretendard"/>', xml)
    return xml


def _postprocess(path: Path):
    """viewProps: snap 끄기, 안내선 비우기. 테마 글꼴 Pretendard."""
    tmp = path.with_suffix(".tmp")
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        names = zin.namelist()
        for item in zin.infolist():
            data = zin.read(item.filename)
            if re.match(r"ppt/theme/theme\d+\.xml$", item.filename):
                data = _theme_fonts(data.decode("utf-8")).encode("utf-8")
            if item.filename == "ppt/viewProps.xml":
                xml = data.decode("utf-8")
                xml = re.sub(r'snapToGrid="\d"', 'snapToGrid="0"', xml)
                xml = re.sub(r'snapToObjects="\d"', 'snapToObjects="0"', xml)
                if "snapToGrid" not in xml:
                    xml = xml.replace("<p:cSldViewPr", '<p:cSldViewPr snapToGrid="0" snapToObjects="0"', 1)
                if "<p:cSldViewPr" not in xml:
                    xml = xml.replace("</p:viewPr>",
                                      '<p:slideViewPr><p:cSldViewPr snapToGrid="0" snapToObjects="0"><p:cViewPr varScale="1">'
                                      '<p:scale><a:sx n="50" d="100"/><a:sy n="50" d="100"/></p:scale><p:origin x="0" y="0"/>'
                                      '</p:cViewPr><p:guideLst/></p:cSldViewPr></p:slideViewPr></p:viewPr>')
                data = xml.encode("utf-8")
            zout.writestr(item, data)
    tmp.replace(path)


def check(path: Path) -> list[str]:
    """저장한 파일 점검: snap 꺼짐, 모든 텍스트 noAutofit, Pretendard 외 글꼴 없음, 18px 미만 글자 없음."""
    probs = []
    with zipfile.ZipFile(path) as z:
        vp = z.read("ppt/viewProps.xml").decode()
        if 'snapToObjects="0"' not in vp or 'snapToGrid="0"' not in vp:
            probs.append("viewProps snap 설정")
        for n in z.namelist():
            if re.match(r"ppt/slides/slide\d+\.xml$", n):
                x = z.read(n).decode()
                nb = x.count("<a:bodyPr")
                na = x.count("<a:noAutofit/>")
                faces = set(re.findall(r'typeface="([^"]+)"', x)) - set(FACE.values())
                small = [int(s) for s in re.findall(r' sz="(\d+)"', x) if int(s) < 900]
                if na < nb:
                    probs.append(f"{n}: noAutofit {na}/{nb}")
                if faces:
                    probs.append(f"{n}: 다른 글꼴 {faces}")
                if small:
                    probs.append(f"{n}: 18px 미만 글자 {sorted(set(small))}")
    return probs


def _count_table_cells(x: str) -> int:
    return x.count("<a:tc>") + x.count("<a:tc ")


# ── HTML 도식 스타일 칩 (회색 입력 · 파랑 모델 · 초록 결과 · 노랑 처리 · 보라 민지 · 빨강 주의) ──
KIND = {  # fill, line, text
    "gray": ("F1F3F6", "D5DAE1", "2E3440"), "white": ("FFFFFF", "D5DAE1", "2E3440"),
    "blue": ("EAF1FF", "6C93F2", "1F3F99"), "green": ("E6F6EE", "56B886", "1E7A46"),
    "amber": ("FFF4DE", "D9A23B", "8A5A00"), "purple": ("F1EBFF", "9B82E6", "5B3FC4"),
    "red": ("FDECEC", "E07A7A", "B83A3A"), "dark": ("14181F", "14181F", "FFFFFF"),
}
for _k, (_f, _l, _t) in KIND.items():
    C.setdefault(f"k-{_k}-fill", _f); C.setdefault(f"k-{_k}-line", _l); C.setdefault(f"k-{_k}-text", _t)


def chip(slide, name, x, y, label, kind="gray", h=44, size=20, weight=600, padx=16, w=None, radius=8,
         line=True, dash=None, align="c"):
    """글자 폭에 맞춘 칩. 반환: (shape, width)."""
    if w is None:
        w = max(text_width(_plain(label), size, weight) + 2 * padx + 8, h)
    s = box(slide, name, x, y, w, h, [(label, {"size": size, "weight": weight, "color": f"k-{kind}-text"})],
            fill=f"k-{kind}-fill", line=f"k-{kind}-line" if line else None, radius=radius, align=align,
            pad=(padx // 2, 0, padx // 2, 0), wrap=False, dash=dash)
    return s, w


def chain(slide, name, x, y, items, h=44, gap=40, size=20, color="stone"):
    """[(label, kind)] 칩을 → 로 잇는다. 반환: 끝 x."""
    for i, (lab, kind) in enumerate(items):
        _, w = chip(slide, f"{name} {i+1}", x, y, lab, kind, h=h, size=size)
        if i < len(items) - 1:
            arrow(slide, f"{name} arrow {i+1}", [(x + w + 8, y + h / 2), (x + w + gap - 8, y + h / 2)],
                  color=color, width_pt=2)
        x += w + gap
    return x - gap


def curve(slide, name, p0, p1, color="stone", width_pt=2.0, head=True, bend=0.5, n=24):
    """가로로 흐르는 S자 곡선 화살표 (3차 베지에 근사)."""
    (x0, y0), (x3, y3) = p0, p1
    x1, x2 = x0 + (x3 - x0) * bend, x3 - (x3 - x0) * bend
    pts = []
    for i in range(n + 1):
        t = i / n
        bx = (1 - t) ** 3 * x0 + 3 * (1 - t) ** 2 * t * x1 + 3 * (1 - t) * t ** 2 * x2 + t ** 3 * x3
        by = (1 - t) ** 3 * y0 + 3 * (1 - t) ** 2 * t * y0 + 3 * (1 - t) * t ** 2 * y3 + t ** 3 * y3
        pts.append((bx, by))
    return arrow(slide, name, pts, color=color, width_pt=width_pt, head=head)


def polygon(slide, name, pts, fill, line=None, line_w=1.5, label=None, size=20, weight=700, color="ink"):
    fb = slide.shapes.build_freeform(E(pts[0][0]), E(pts[0][1]), scale=1.0)
    fb.add_line_segments([(E(px), E(py)) for px, py in pts[1:]], close=True)
    s = fb.convert_to_shape()
    s.name = name
    s.fill.solid(); s.fill.fore_color.rgb = rgb(fill)
    if line:
        s.line.color.rgb = rgb(line); s.line.width = E(line_w)
    else:
        s.line.fill.background()
    _drop_style(s)
    tf = s.text_frame
    _no_autofit(tf, 4, 0, 4, 0, anchor="m")
    if label:
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        add_runs(p, label, size, weight, color)
    return s
