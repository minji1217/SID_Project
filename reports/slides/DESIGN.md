---
# 랏코(Rakko) 16:9 슬라이드 디자인 시스템 — 디자인 토큰
# 색 값 출처: 기준 덱(NESCL 논문 리뷰 pptx)에서 뽑은 값 = (ref)
# 기준 덱에 없어 같은 계열로 정한 값 = (set)
canvas: { width: 1920, height: 1080 }   # pptx 13.333in × 7.5in, 1px = 0.5pt
colors:
  primary: "#14181F"          # (ref) 장 번호 칩 배경
  on-primary: "#FFFFFF"
  on-dark: "#FFFFFF"
  ink: "#14181F"              # (ref) 제목
  ink-strong: "#000000"
  charcoal: "#2E3440"         # (ref) 본문
  slate: "#5B6577"            # (ref)
  steel: "#5B6577"            # (ref) 장 이름, 부제
  stone: "#8A93A3"            # (ref)
  muted: "#8A93A3"            # (ref) 쪽 번호, 꼬리말
  canvas: "#FFFFFF"
  surface: "#F4F6F9"          # (set)
  surface-soft: "#F8F9FB"     # (set)
  hairline: "#D5DAE1"         # (set)
  hairline-soft: "#E3E6EB"    # (ref) 머리말·꼬리말 선
  brand-blue: "#4C7BF4"       # (ref) 목차 숫자, 강조 테두리
  brand-blue-deep: "#1F3F99"  # (ref) 번호 소제목, 강조 글자
  brand-blue-700: "#2F55C8"   # (set)
  brand-blue-200: "#DCE6FD"   # (set) 파스텔 블루 카드·표 강조
  brand-cyan: "#A9C8FB"       # (set)
  brand-purple: "#6E56CF"     # (ref, 테마)
  brand-lavender: "#ECE8FA"   # (set) 파스텔 라벤더 카드
  brand-coral: "#F2705E"      # (set) 기본 사용 안 함
  brand-magenta: "#D3468C"    # (set) 기본 사용 안 함
  amber: "#F2B33D"            # (set) 별표 하나
  success-bg: "#E3F5EA"       # (set)
  success-text: "#1E7A46"     # (set)
  error: "#D45656"
typography:   # px (1920 캔버스). pptx pt = px / 2
  cover-title:   { size: 112, weight: 800, lh: 1.10, ls: -0.03em }
  section-title: { size: 96,  weight: 800, lh: 1.10, ls: -0.03em }
  stat-giant:    { size: 160, weight: 900, lh: 1.00, ls: -0.03em }
  stat-lg:       { size: 72,  weight: 800, lh: 1.05, ls: -0.02em }
  title:         { size: 56,  weight: 700, lh: 1.20, ls: -0.02em }
  subtitle:      { size: 28,  weight: 500, lh: 1.40, ls: -0.01em }
  chapter:       { size: 22,  weight: 600, lh: 1.30 }
  heading-lg:    { size: 40,  weight: 700, lh: 1.25, ls: -0.02em }
  heading-md:    { size: 32,  weight: 700, lh: 1.30, ls: -0.01em }
  card-title:    { size: 26,  weight: 600, lh: 1.35 }
  body-lg:       { size: 28,  weight: 400, lh: 1.50 }
  body:          { size: 24,  weight: 400, lh: 1.50 }
  body-strong:   { size: 24,  weight: 700, lh: 1.50 }
  table:         { size: 22,  weight: 400, lh: 1.45 }
  caption:       { size: 20,  weight: 400, lh: 1.50 }
  footer:        { size: 18,  weight: 400, lh: 1.40 }
  badge:         { size: 18,  weight: 600, lh: 1.30 }
spacing: { xxs: 8, xs: 12, sm: 16, md: 24, lg: 32, xl: 48, xxl: 64, margin: 96 }
rounded: { xs: 4, sm: 6, md: 8, lg: 12, xl: 16, xxl: 20, hero: 32, full: 9999 }
---

# 랏코 슬라이드 디자인 시스템

> 생성 스크립트: `reports/slides/build_pipeline_deck.py`(공통 헬퍼 포함). 폰트: Pretendard 1.3.9 static OTF (npm `pretendard`, OFL).
> pptx 폰트 이름은 OTF의 legacy family 이름을 쓴다: 400 `Pretendard`, 500 `Pretendard Medium`, 600 `Pretendard SemiBold`, **700 `Pretendard` + 굵게(b=1)**, 800 `Pretendard ExtraBold`, 900 `Pretendard Black`.
> **사용자 선호: 실제 글자는 토큰보다 한 단계 두껍게** (400→500, 500→600, 600→700, 700→800, 800→900). `rakko.py`의 `WEIGHT_UP`.
> (`Pretendard Bold`라는 이름은 Windows·LibreOffice에서 찾지 못해 다른 글꼴로 바뀐다.) 보는 컴퓨터에 Pretendard 9개 굵기가 설치돼 있어야 한다.

## Overview

This is the design system for **16:9 presentation slides** of the brand **랏코 (Rakko)**. It governs every slide Claude produces: layout, typography, color, components, and brand usage. Output is slides only: no web pages, no documents, no other aspect ratios.

The visual voice is confident, technical, almost editorial: a stark white canvas with deep-black typographic emphasis, broken open by saturated accent-color cards used sparingly. Slides are information-dense but orderly: every content slide is built on the same fixed frame so that the chapter label, title, subtitle and page number never move from page to page.

**Key Characteristics:**
- **16:9 only.** Canvas is exactly 1920 × 1080 px. Never produce 4:3, 1:1, 9:16 or any other ratio, and never ask which ratio to use.
- **Pretendard only.** One typeface family, weights 100–900, for all text. The single exception is mathematical formulas, which are placed as LaTeX bold images (see Typography → Formulas).
- **Fixed frame.** Chapter label, title, subtitle, and page number occupy identical coordinates on every content slide (see Slide Frame).
- **Dense but airy.** Bodies carry meaningful content with deliberate, generous whitespace, measured from the user's LightGCN deck: a content block of about 55–75% of the body height, centered slightly below the middle (see Whitespace & Placement Spec), never crammed and never carelessly empty.
- **No logo.** Slides carry no logo, wordmark, or placeholder box anywhere. The top-right corner of the frame is left clean: no logo, no pills, no date or tag.
- Stark monochrome palette — black ({colors.primary}) and white ({colors.canvas}) — with saturated accent colors reserved for emphasis moments.
- **Blue / pastel palette by default.** Unless the user asks otherwise, build slides in blues and pastel blue-violet tints (deep navy for emphasis, pastel blue and lavender for cards and charts); keep coral/magenta out unless explicitly requested.
- **Diagrams follow the supplied figure.** When the user provides a figure or sketch, reproduce its structure and content faithfully as editable native shapes, at a modest size (see Diagrams).
- Pill-shaped tags, badges and tabs ({rounded.full}); cards in two families: quiet white cards (16px radius) and vibrant accent cards (32px radius).

## Colors

> Token values are defined in the design tokens, not repeated here. Always reference tokens, never raw hex values (the only exception is the error red `#d45656`).

### Brand & Accent
- **Brand Coral** ({colors.brand-coral}): Highest-impact accent. Key numbers, "NEW" badges, one featured card per slide at most.
- **Brand Magenta** ({colors.brand-magenta}): Secondary accent for a second category in a comparison.
- **Brand Blue** ({colors.brand-blue}): Primary blue accent; third category, chart series.
- **Brand Blue Deep** ({colors.brand-blue-deep}): Links, active states, emphasis text on light surfaces.
- **Brand Blue 700** ({colors.brand-blue-700}): Tag and reference text color.
- **Brand Cyan** ({colors.brand-cyan}): Light blue for flat tints and chart series; never for gradients.
- **Brand Blue 200** ({colors.brand-blue-200}): Info-tag and code-badge backgrounds.
- **Brand Purple** ({colors.brand-purple}): Fourth category.

Accent colors encode **categories or emphasis**, not decoration. Assign one accent per category and keep that assignment consistent across the whole deck.

### Surface
- **Canvas White** ({colors.canvas}): Slide background and card surface.
- **Surface** ({colors.surface}): Subtle section backgrounds, takeaway bars, table headers.
- **Surface Soft** ({colors.surface-soft}): Quieter section divisions.
- **Hairline** ({colors.hairline}): 1px card border and primary divider.
- **Hairline Soft** ({colors.hairline-soft}): Table-row dividers, footer rule.

### Text
- **Ink** ({colors.ink}): Titles, primary text.
- **Ink Strong** ({colors.ink-strong}): Pure black, for maximum contrast on small text.
- **Charcoal** ({colors.charcoal}): Body text on light surfaces.
- **Slate** ({colors.slate}): Secondary text, metadata.
- **Steel** ({colors.steel}): Subtitles, chapter label, table headers.
- **Stone** ({colors.stone}): Captions and inactive labels.
- **Muted** ({colors.muted}): Footer text, de-emphasized labels.

### Semantic
- **Success Background** ({colors.success-bg}) / **Success Text** ({colors.success-text}): Positive-status badges.
- Error: `#d45656` for negative or warning states only.

## Typography

### Font Family — Pretendard ONLY
**Pretendard is the only typeface allowed for text**, for every text element on every slide (mathematical formulas are the one exception, see Formulas below): titles, body, tables, chart labels, axis ticks, badges, footers, page numbers, captions, and code-like text. Never use, suggest, or fall back to DM Sans, Inter, Helvetica, Arial, Noto Sans, system fonts, or any monospace font. For code-like or technical strings, keep Pretendard and mark them with a {colors.surface} background chip and {rounded.xs} corners instead of a monospace face.

Use the static OTF files, one file per weight:

| Weight | File |
|---|---|
| 100 | Pretendard-Thin.otf |
| 200 | Pretendard-ExtraLight.otf |
| 300 | Pretendard-Light.otf |
| 400 | Pretendard-Regular.otf |
| 500 | Pretendard-Medium.otf |
| 600 | Pretendard-SemiBold.otf |
| 700 | Pretendard-Bold.otf |
| 800 | Pretendard-ExtraBold.otf |
| 900 | Pretendard-Black.otf |

Declare every weight explicitly and embed the files so the deck renders identically anywhere:

```css
@font-face { font-family: "Pretendard"; font-weight: 100; src: url("Pretendard-Thin.otf") format("opentype"); }
@font-face { font-family: "Pretendard"; font-weight: 200; src: url("Pretendard-ExtraLight.otf") format("opentype"); }
@font-face { font-family: "Pretendard"; font-weight: 300; src: url("Pretendard-Light.otf") format("opentype"); }
@font-face { font-family: "Pretendard"; font-weight: 400; src: url("Pretendard-Regular.otf") format("opentype"); }
@font-face { font-family: "Pretendard"; font-weight: 500; src: url("Pretendard-Medium.otf") format("opentype"); }
@font-face { font-family: "Pretendard"; font-weight: 600; src: url("Pretendard-SemiBold.otf") format("opentype"); }
@font-face { font-family: "Pretendard"; font-weight: 700; src: url("Pretendard-Bold.otf") format("opentype"); }
@font-face { font-family: "Pretendard"; font-weight: 800; src: url("Pretendard-ExtraBold.otf") format("opentype"); }
@font-face { font-family: "Pretendard"; font-weight: 900; src: url("Pretendard-Black.otf") format("opentype"); }

body { font-family: "Pretendard", sans-serif; word-break: keep-all; overflow-wrap: break-word; font-feature-settings: "tnum" 1; }
```

The trailing `sans-serif` is only the CSS keyword required for syntax; it is never an intended font. If a Pretendard file fails to load, stop and report it instead of silently rendering another face. In PowerPoint/Keynote-style exports, set every text run to the "Pretendard" family (with the matching weight style) and never leave a default font on any run.

### Hierarchy (1920 × 1080 canvas, sizes in px)

| Token | Size | Weight | Line Height | Letter Spacing | Use |
|---|---|---|---|---|---|
| `{typography.cover-title}` | 112 | 800 | 1.10 | -0.03em | Cover slide title |
| `{typography.section-title}` | 96 | 800 | 1.10 | -0.03em | Section-divider slide title |
| `{typography.stat-giant}` | 160 | 900 | 1.00 | -0.03em | One hero number on a stat slide |
| `{typography.stat-lg}` | 72 | 800 | 1.05 | -0.02em | KPI numerals in cards and strips |
| `{typography.title}` | 56 | 700 | 1.20 | -0.02em | **Slide title (fixed position, one line)** |
| `{typography.subtitle}` | 28 | 500 | 1.40 | -0.01em | **Slide subtitle (fixed position, one line)** |
| `{typography.chapter}` | 22 | 600 | 1.30 | 0 | **Chapter label (fixed position)** |
| `{typography.heading-lg}` | 40 | 700 | 1.25 | -0.02em | In-body headline moments |
| `{typography.heading-md}` | 32 | 700 | 1.30 | -0.01em | Column / panel headers |
| `{typography.card-title}` | 26 | 600 | 1.35 | 0 | Card titles |
| `{typography.body-lg}` | 28 | 400 | 1.50 | 0 | Lead body, takeaway bar |
| `{typography.body}` | 24 | 400 | 1.50 | 0 | Primary body text |
| `{typography.body-strong}` | 24 | 700 | 1.50 | 0 | Body emphasis |
| `{typography.table}` | 22 | 400 | 1.45 | 0 | Table cells, chart labels |
| `{typography.caption}` | 20 | 400 | 1.50 | 0 | Notes, legends, sources |
| `{typography.footer}` | 18 | 400 | 1.40 | 0 | Footer text, page number — **absolute minimum size on any slide** |
| `{typography.badge}` | 18 | 600 | 1.30 | 0 | Tags, badges, pills |

### Formulas (the only exception to Pretendard)
Mathematical formulas are not typed as text. They are placed as **LaTeX images in the paper style, set fully in bold** (Computer Modern math: bold vectors and matrices such as **e**, **W**, bold Σ and brackets, italic bold Greek letters), because Pretendard lacks many math symbols (∈, ⊙, ∑ limits, 𝒩, ŷ).
- **Rendering:** build each formula with `pdflatex` using the `standalone` class, `amsmath`, `amssymb`, `bm`, `xcolor`, wrapped as `{\boldmath ... }`, and convert it with `pdftocairo -png -transp -r 576`. Output is a tightly cropped **transparent PNG**, one image per formula (or per aligned block of lines).
- **Scale:** the PNG is 12 pt text at 576 dpi, so 1 PNG px = 0.25 slide px at 24 px text size. Multiply by 1.0–1.3 to reach the intended formula size; formulas should read at roughly 24–30 px, never below 20 px.
- **Color:** the formula color follows its container ({colors.ink} on white cards, or a dark brand text color on pastel cards); terms being highlighted, such as removed terms, use `\textcolor` in the highlight color (violet {colors.brand-purple}) and stay bold.
- **Alt text:** every formula image carries an alt text describing it, and the formula is also written in plain text in the speaker notes.
- **Words stay Pretendard:** all labels, captions and explanations around a formula remain Pretendard text. Never put Korean prose inside a formula image.
- Do not use any other font (including Cambria Math) for formulas, and do not type formulas as text runs.

### Principles
- **Weight discipline:** 400 body · 500 subtitles and secondary emphasis · 600 labels and card titles · 700 titles and strong emphasis · 800–900 only for cover/section titles and large numerals. Weights 100–300 are allowed only at 64px and above as decorative display text, never for body copy.
- **Never below 18px.** If text does not fit, cut the copy or restructure the layout; do not shrink the type.
- **Korean line breaking:** always `word-break: keep-all` so words are not split mid-syllable-block; avoid orphaned single characters on the last line.
- **Mixed Korean/English:** keep both in Pretendard at the same size and weight; do not change fonts for Latin text.
- **Numerals** use tabular figures (`"tnum"`) in tables, charts and KPI strips so columns align.
- Emphasis comes from weight and color, never italics (Pretendard has no italic).

## Layout

### Canvas
- Exactly **1920 × 1080 px (16:9)**, one slide per page, no scrolling, no responsive variants.
- Background is always {colors.canvas} (white) on every slide, including cover, section dividers and closing. Never use dark or colored full-slide backgrounds.

### Spacing System
- **Base unit**: 8px.
- **Tokens**: `{spacing.xxs}` 8 · `{spacing.xs}` 12 · `{spacing.sm}` 16 · `{spacing.md}` 24 · `{spacing.lg}` 32 · `{spacing.xl}` 48 · `{spacing.xxl}` 64 · `{spacing.margin}` 96.
- **Card padding**: 32–40px for all cards.
- **Gaps between modules**: 32–48px (cards, columns, rows); 8–16px inside a group.

### Slide Frame (IDENTICAL on every content slide)

All coordinates are in px on the 1920 × 1080 canvas; left and right margins are 96px (content width 1728px). **Do not move, resize, restyle, or omit any frame element from slide to slide.**

| Element | Position | Spec |
|---|---|---|
| **Chapter label** | x 96 · y 48 · h 40 · left-aligned | Number chip (pill, {colors.primary} background, white text, `{typography.badge}`, e.g. "02") followed by the chapter name in `{typography.chapter}` {colors.steel}. Same chapter name on every slide of that chapter. |
| **Title** | x 96 · y 112 · w 1728 · h 72 | `{typography.title}` {colors.ink}. **One line only.** Roughly 26 Korean or 48 Latin characters max; shorten the wording rather than wrap or shrink. |
| **Subtitle** | x 96 · y 192 · w 1728 · h 40 | `{typography.subtitle}` {colors.steel}. One line only, one clarifying sentence that supports the title. |
| **Header rule** | y 256 · full content width | 1px {colors.hairline-soft}. |
| **Body zone** | x 96 · y 288 → 960 · w 1728 · h 672 | All slide content lives here, using about 55–75% of its height with deliberate whitespace (see Whitespace & Placement Spec). |
| **Footer rule** | y 984 · full content width | 1px {colors.hairline-soft}. |
| **Footer left** | x 96 · y 996 · h 28 | Source / note line in `{typography.footer}` {colors.muted}; deck name if no source. |
| **Footer right** | right edge x 1824 · y 996 · h 28 | Page number as "03 / 24" in `{typography.footer}` {colors.muted}, tabular figures. |

Rules for the frame:
- Title and subtitle are always present on content slides; the subtitle is never dropped.
- Chapter label, title and subtitle must start at exactly the same x/y on every slide, so flipping through the deck shows them perfectly stationary.
- Nothing from the body may intrude into the header or footer bands; nothing from the frame may be repositioned to "make room".

### Special Slide Types
Cover, section-divider and closing slides may use their own composition, but they still use the same 96px margins. Section dividers show the chapter number and chapter name at the same left x (96) with `{typography.section-title}`.

- **Cover:** `{typography.cover-title}` left-aligned at x 96, subtitle below it in `{typography.body-lg}`, presenter/date row anchored to the bottom margin (y 960). Keep it text-only on white: no figure, diagram, card or gradient on the cover. Enlarge the title (two lines) so the page feels balanced, and let whitespace do the rest.
- **Section divider:** large chapter number in `{typography.stat-giant}`, chapter title in `{typography.section-title}`, and a 3–5 item list of what the chapter covers so the slide is not sparse.
- **Closing:** plain white, text-first, same style as content slides: a takeaway title, 3–4 key conclusions as bullets, and an optional "next reading" list set close beside them (not a separate colored card). Put code links in the footer source line. Never put a personal email address or contact details on a slide unless the user asks. See Patterns Learned from Reference Decks for the takeaways slide and the thank-you slide.

### Body Grid
- 12-column grid across 1728px with 24px gutters. Common splits: 12 (full), 6 + 6, 4 + 4 + 4, 8 + 4, 3 × 4, 4 × 3.
- Cards in the same row always share the same height; columns share the same top and bottom edges.

### Density & Whitespace Rules

Slides are **dense but airy**: whitespace is a designed element, not leftover space. The goal is a calm, editorial page in which every empty area is deliberate and visibly framed by the content around it.

1. **Follow the Whitespace & Placement Spec.** The content block covers about 55–75% of the body zone height (y 288 → 960) and 60–90% of its width, starts 80–160px below the header rule, and may end 70–180px above the footer rule. It must never feel crammed, and never leave a large unexplained hole.
2. **Purposeful whitespace only.** Empty space is allowed when it (a) separates groups, (b) gives a hero element (diagram, number, chart) room to breathe, or (c) is part of a clear asymmetric layout. Whitespace is never a blank column or an empty bottom third beside crowded content.
3. **Generous spacing.** Gaps between modules are **32–48px**; card padding **32–40px**; line height 1.5 for body text. Group related items tightly (8–16px) and separate groups clearly, so the eye reads structure from spacing instead of borders.
4. **Fewer, larger units.** Carry **2–4** information units per slide (hard maximum 5). If more are needed, split the slide. Prefer one strong visual (diagram, chart, hero stat) plus one or two supporting cards over many small boxes.
5. **Text before boxes.** Never put plain sentences or bullets inside cards by default (see Text-First Layout).
6a. **Cut before you shrink.** Keep 1–2 short bullets per card (one line each where possible), and drop secondary formulas, captions and repeated explanations rather than reducing type below 18px. A formula or fact that already appears once on the slide is not repeated.
6. **Anchor row is optional and normally absent.** By default a slide has no bottom line at all. Only when the content calls for it, add one plain text line at about y 880–940: a takeaway, or a "→" hand-off line (see Story continuity). Never use a KPI strip together with it.
7. **Balance.** Distribute visual weight across the width; left and right groups should feel equal in weight, with the empty space shared evenly rather than pooled on one side.
8. **Margins stay.** Content never touches the 96px side margins, the header rule, or the footer rule.

## Text-First Layout (reference density)

The target look is a calm lecture slide: a one-line title, a hairline rule, then **plain text and figures on white**, with generous margins and the footer below. Think "a few clear bullet lines plus one or two figures with short captions".

- **Text is not boxed.** Body copy, bullets, step headings and takeaways are set as plain text on the white canvas: a blue ({colors.brand-blue-deep}) bullet dot, 22–26px text in {colors.charcoal}, 8–16px between lines. Do **not** wrap every sentence or step in a card, pill or filled bar.
- **Structure through typography, not containers.** Use numbered headings ("1. …", "2. …") in {typography.card-title} {colors.brand-blue-deep}, weight and color contrast, spacing, and thin hairline rules. Curved arrows or short connectors may link steps.
- **Boxes are rare.** Use at most one filled card or highlight per slide, only for a single key number, one callout, or one table. Figures, diagrams and charts sit directly on the canvas; a figure never needs a surrounding card.
- **Figure + caption.** Place each figure with a short one-line caption or numbered label above it (for example "2. 단어 사이의 관계를 파악"), left-aligned to the figure; arrange two or three figures in a loose row with wide gaps.
- **Takeaway as a sentence.** A closing takeaway is one plain text line (blue "핵심" label + sentence), not a filled bar.
- **Density target.** About 3 bullet lines or 3 short step blocks plus 1–3 figures; text occupies well under half of the body area, the rest is figure and whitespace.

## Whitespace & Placement Spec (measured from the LightGCN reference deck)

Measured on the user's own 31-page LightGCN review (converted to the 1920 × 1080 canvas). These numbers define "how much whitespace" and "where things go"; they override looser wording elsewhere. Our frame margin is 96px, which sits inside the measured 88–110px range.

### Measured facts
| Property | Measured value |
|---|---|
| Left margin of title, subtitle and left text | 88–110px (title and subtitle start at the same x) |
| Right margin of content | 40–110px; most figures stop at x 1750–1850 |
| Header band (title + subtitle) | y 100 → ~340, about 30% of the slide height |
| First body element starts | y 340–440, i.e. 40–100px below the subtitle |
| Body content bottom | y 900–1030; text-only slides stop near y 900–950, leaving 130–180px empty at the bottom |
| Content block size (text and figures together, header excluded) | 46–66% of slide height, 60–90% of slide width |
| Content vertical center | y 590–700 (55–65% of slide height): content sits slightly below the middle, never hugging the header |
| Gap between a text block and a figure side by side | 60–115px |
| Gap between stacked text blocks | 40–125px; list rows (agenda) repeat every 125px |
| Text block width | 750–850px, about 40–45% of slide width |
| Side-by-side figure width | 30–48% of slide width (about 585–900px); the architecture figure is only ~30% wide (585 × 260px) |
| Ink coverage (non-white pixels) | text-only slides 2–6%; figure slides 8–12%; only full code-panel slides exceed 30% |

### Placement rules
1. **Header first, then air.** After the header rule leave 80–160px before the first body element (first element top at y 336–416). Never start the body flush under the rule.
2. **Two blocks, center-aligned.** On a text + figure slide, make two blocks: a text block (640–800px wide) and a figure (520–840px wide) with a 60–110px gap. Center them on a shared horizontal center line at about y 600–680 instead of top-aligning them; their heights may differ.
3. **Z-pattern, staggered.** Reading order is title → left text → right figure → lower-right note. The figure may rise to the title band and sit right-aligned with a secondary text block under it, while the main steps stay lower-left. Leave at least one quadrant clearly empty; never fill all four.
4. **Content block ≈ 55–75% of the body zone height** (about 370–500px of the 672px body zone) and 60–90% of the width. Short slides (one formula, one table) use a centered block 60–66% wide at vertical center y 640–680.
5. **Bottom air.** Text-only slides end 100–180px above the footer rule; figure slides may end at y 960. Never pad the bottom with extra boxes to fill it.
6. **Vertical rhythm.** Heading-to-first-bullet 16–24px; bullet-to-bullet 16–24px; section-to-section 48–96px; single-line list rows (agenda, takeaways) repeat on a 120–130px pitch.
7. **Text blocks stay narrow.** Body text never runs wider than about 850px; long lines are broken by the layout, not by shrinking type. A paragraph plus figure should leave the figure's side visibly emptier than the text side.
8. **Paper thumbnail pattern.** A tall figure (paper first page, 480 × 620px) sits left at x 96–580 and a short text block starts at x ~660, vertically centered on the figure.
9. **Code panel pattern.** The code panel takes the right ~60% (x ≥ 660) at nearly full height (y 90–1020); the left column holds only the heading, the subtitle and one or two caption lines anchored at y 690–910.
10. **Density check.** Count non-white pixels: text-only slide above 6% or figure slide above 12% means too dense; remove or shrink something before adding anything.
11. **Agenda slide grid.** Two columns starting at x 132 and x 945, first row at y 364, row pitch 125px, big blue numerals followed by the chapter name; roughly 40% of the slide stays empty.

## Patterns Learned from Reference Decks

Distilled from the user's own decks (a 15-page lecture-review deck and a paper-review deck on LightGCN). Apply these whenever the matching slide type appears.

### Deck structure
- **Cover:** large left-aligned title, no figure; a small meta row anchored at the bottom with label-over-value pairs (for example "작성자 / 발표일 / 참고 자료"). Labels in {colors.muted} 18px, values in {colors.ink} 22px weight 600.
- **Agenda (Contents) slide:** one slide after the cover. Chapters are listed as columns or a two-column numbered list with big blue numerals ("01", "02" …), each with the chapter name, a one-line guiding question, 2–4 sub-topics and a page range. Chapters covered by the current deck part may be marked with a blue top rule; all others stay neutral. Add one line below with the reference/source.
- **Chapter flow:** every content slide carries the chapter chip (number) and chapter name from the agenda, so the listener always knows where they are. Page numbers are "NN / total".
- **Story continuity (optional, content-driven):** by default slides end without any connecting sentence. Add a transition line only when the content itself raises an open question or sets up the next slide, and only on that slide. The line is one plain-text sentence at the bottom of the body zone, starting with a small blue arrow "→" (for example "→ 은닉층의 가중치는 어떻게 고칠까?"). Never add it to every slide as a routine, never add it to a slide where it would be filler, and omit it on cover, agenda, closing and thank-you slides.
- **Closing:** a "느낀 점 / takeaways" slide with three plain columns separated by thin top rules (blue on the first, hairline on the others), each with a blue numbered heading and 2–3 lines; then a separate thank-you slide: the sentence "들어주셔서 감사합니다" as title, the deck name as subtitle, and the source credits in the footer. No gradients, cards or decorative graphics on either.

### Content slide template
- Claim title (one full sentence, subject + predicate: "CNN은 필터로 이미지의 특징을 뽑아낸다"), one-line subtitle that adds the condition or example, hairline rule.
- Left: 2–3 numbered sections. Each section is a blue heading "1. …" (card-title size) followed by 1–3 plain bullets with the key term in bold. Right: one figure with a short numbered caption. Alternative: 3 equal numbered columns, each with a small figure on top and text below.
- A paper-style subsection label (for example "Embedding Propagation Rule", "Matrix Form", "Model Training") may sit under the title in {colors.brand-blue-deep} when a chapter has several slides on one topic.

### Figures, tables and equations
- **Use the source's own figures** (paper figures, tables, training curves, textbook diagrams) at readable size with the original caption or figure number kept; crop tightly, never stretch. Redraw only when the user asks or when no source figure exists.
- **Annotate figures** with at most one amber star plus a short blue callout that states the point ("오히려 성능이 증가한 결과"), and with thin blue rounded outlines around the exact term, line or cell being discussed.
- **Result tables:** put the claim as a blue headline above the table ("NGCF보다 LightGCN의 성능이 좋음"); keep the paper's best numbers bold; highlight the key column or row with a light blue tint; never restyle the numbers.
- **Equation slides:** show the equation large (bold LaTeX image), then add numbered blue callouts below or beside it ("1. feature transformation 제거 → 가중치 행렬 W 없음"), each pointing to its term with a thin rounded outline around that term. When a model is simplified, show before → after equations with a chevron arrow (">>>") between them.
- **Process / flow diagrams:** a horizontal chain of small neutral chips with arrows, key steps filled light blue or blue, shapes (feature maps, matrices) drawn flat between chips, a return lane with a long arrow for backward flows, and short labels above and below.
- **Multi-figure rows:** 2–3 figures in a loose row, each with a numbered blue caption ("1. 직선 하나로는 나눌 수 없다"), equal height, generous gaps.

### Code-review slides
- Show the code as a screenshot or flat dark code panel at large size on the right two-thirds, with a blue chapter-style heading and the file/function name as subtitle on the left (for example "model.py – LightGCN class → 임베딩 초기화"). This is the only place where a dark panel is allowed, and the slide background stays white.
- **Spotlight technique:** repeat the same code across several slides; on each, dim everything except the 2–6 lines under discussion (semi-transparent gray overlay with the highlighted lines kept at full contrast), and put one or two plain bullets as the caption on the left, anchored to the lower left.
- Code links and repository names go in the subtitle or footer, never in a card.

### Tone and restraint
- One accent hue (blue) carries every emphasis; bold plus blue is the only emphasis style. No icons, no decorative shapes.
- Text stays plain: sentences with bold key terms; hierarchy through numbered blue headings, not boxes.
- Hand-drawn style curved black arrows may link sub-points on text-heavy slides.

## Diagrams

Applies when the user supplies a figure, sketch or reference slide, or asks for an architecture / flow / mechanism diagram.

- **Faithful structure.** Keep the supplied figure's nodes, groups, labels, order and connections exactly; do not simplify away elements. Redraw them as native, editable shapes (one grouped "Diagram" object where possible), not as a pasted screenshot, and recolor to the deck palette.
- **Figure size.** A diagram or figure is a supporting visual, not the whole slide: it spans at most about 60–65% of the body width and about 75% of the body height, centered vertically against the text beside it, with at least 48px of clear space around it. Inside it, text stays at 16px or larger, so shrink the geometry and line weights rather than the type. If it still feels large, cut parts of the figure's secondary labels rather than enlarging the canvas it sits on.
- **Free placement.** Inside the body zone, modules (cards, step boxes, formula chips, callouts) are not forced onto a rigid grid: stagger x positions and heights, let a diagram, a note card and a formula chip sit at different offsets, as long as gaps stay 24px or more, nothing overlaps, and the body still reaches y 960. The Slide Frame itself never moves.
- **Arrow style.** Flow arrows inside diagrams are thick (about 3pt) light-slate lines with solid triangle heads; loop or iteration arrows are dotted amber curves with a triangle head; arrows between sequential step texts are short hand-drawn style curved black arrows (about 56px wide and 46px tall, 2.5pt line, triangle head) that bulge to the left and point down-right at the next step heading, placed in the left margin gap between steps. Draw arrowheads as part of the line, never as separate text glyphs.
- **Callouts.** A highlighted label may sit on a pastel highlight bar with a small amber star; keep it to one per diagram.
- **Labels.** Node labels, captions and sub/superscripted symbols follow the Formulas rule (bold LaTeX image) when they contain math; plain words stay Pretendard at 16px or larger inside diagrams.

## Editability Settings (the user edits every deck afterwards)

The user will move, resize and rewrite objects after delivery, so the file must behave predictably under the mouse. Apply every item below to each generated file.

- **No snapping.** Turn off snap-to-objects, snap-to-grid and guides in the file's view settings (`ppt/viewProps.xml`: `snapToObjects="0"`, `snapToGrid="0"`, empty guide list). Generators ship `snapToObjects="1"` by default, which pulls dragged shapes onto the edges of neighbouring shapes and makes them jump to places the user did not choose. Do not add alignment guides, distribution rules or magnetic layouts.
- **No autofit.** Every text body carries an explicit "do not autofit" setting (`<a:noAutofit/>`), so boxes never resize or shrink text on their own when words are added or removed.
- **Independent, plain objects.** Each text block is its own native text box with a fixed position and size; each card, line, arrow and shape is a separate native object that can be moved on its own. Do not group objects, except one diagram that the user explicitly wants to move as a unit, and name that group.
- **Native text only.** Titles, bullets, labels and table text are real text, never images. Bullets are native paragraph bullets in one text box per list (one paragraph per item), so items can be added, deleted or reordered with the keyboard. Formulas are the only images (see Formulas).
- **Absolute positions, no hidden layout.** Place objects with explicit coordinates on the blank layout; do not rely on layout placeholders, inherited positions, centering helpers or "distribute" arrangements that reflow when one object moves. Keep a small safety margin inside each text box (padding) instead of tight boxes.
- **Meaningful names.** Name every object by its role ("Title", "Subtitle", "Summary list", "Step arrow 2") so the selection pane is readable.
- **Verify before delivery.** After writing the file, check the view settings and text-body settings above, validate that the file opens, and render it once to confirm nothing moved.

## Elevation & Depth

Slides are flat. Default is no shadow with a {colors.hairline} border.

| Level | Treatment | Use |
|---|---|---|
| 0 (flat) | No shadow; `1px solid {colors.hairline}` | Default cards, table rows |
| 1 (subtle) | `rgba(0, 0, 0, 0.04) 0px 1px 2px 0px` | Featured quiet card (rare) |

- Vibrant accent cards, when used, are flat solid fills; no shadow.
- No glows, no blurs, no 3D, no drop shadows on text.

## Shapes

### Border Radius Scale

| Token | Value | Use |
|---|---|---|
| `{rounded.xs}` | 4px | Code-like chips |
| `{rounded.sm}` | 6px | Table cells, compact chips |
| `{rounded.md}` | 8px | Small panels |
| `{rounded.lg}` | 12px | Compact cards |
| `{rounded.xl}` | 16px | Standard quiet cards, takeaway bars |
| `{rounded.xxl}` | 20px | Larger feature panels |
| `{rounded.hero}` | 32px | Vibrant accent cards, featured callouts |
| `{rounded.full}` | 9999px | All pills, tags, badges, number chips |

The radius contrast between 32px accent cards and 16px quiet cards on the same slide is a signature of the system. Charts and images use the radius of the card that contains them; no rounded internal frames inside cards.

## Components

> Slides are static: no hover, focus, navigation, form or input components. Do not draw buttons, search bars, nav bars, cursors or browser chrome.

### Cards & Containers

**`card-base`** — Standard content card.
- Background {colors.canvas}, rounded {rounded.xl}, padding {spacing.md}–{spacing.lg}, border `1px solid {colors.hairline}`.
- Title `{typography.card-title}`, body `{typography.body}` or `{typography.table}`.

**`card-feature`** — Quieter panel on gray.
- Background {colors.surface}, rounded {rounded.xl}, padding {spacing.lg}.

**`card-accent`** — Vibrant emphasis card (use at most one or two per slide).
- Background one of {colors.brand-coral}, {colors.brand-magenta}, {colors.brand-blue}, {colors.brand-purple}; text {colors.on-dark}; rounded {rounded.hero}; padding {spacing.lg}.
- Hosts a headline number (`{typography.stat-lg}`) or one short statement (`{typography.heading-md}`) with a supporting line at 80% white opacity.

**`takeaway-bar`** — Zone B anchor.
- Background {colors.surface}, rounded {rounded.xl}, height 96px, padding 0 {spacing.lg}; badge pill + `{typography.body-lg}` sentence, vertically centered.

**`kpi-strip`** — Zone B anchor.
- 3–4 equal cells, 96px tall, {rounded.xl}, separated by 24px gaps; each cell has the number in `{typography.stat-lg}` {colors.ink} (or white on an accent cell) and the label in `{typography.caption}` {colors.steel}.

### Tags, Badges & Pills

**`chapter-chip`** — Chapter number in the frame: {colors.primary} pill, white `{typography.badge}` text, height 32px, padding `0 14px`.

**`badge-new`** — {colors.brand-coral} background, {colors.on-dark} text, `{typography.badge}`, {rounded.full}, padding `4px 12px`.

**`badge-beta` / `badge-info`** — {colors.brand-blue-200} background, {colors.brand-blue-deep} text, `{typography.badge}`, {rounded.full}.

**`badge-success`** — {colors.success-bg} background, {colors.success-text} text, `{typography.badge}`, {rounded.full}.

**`pill-tab` / `pill-tab-active`** — Segment labels inside a slide (e.g., period or category selectors drawn as static pills): inactive = white with hairline border and {colors.steel} text; active = {colors.primary} with {colors.on-primary} text. Always {rounded.full}.

### Data Tables

**`data-table`** — Background {colors.canvas}, rounded {rounded.md}, border `1px solid {colors.hairline}`, text `{typography.table}`.
- **Header row:** {colors.surface} background, {colors.steel} text, 600 weight, row height 56px.
- **Body rows:** row height 56–64px (stretch row height so the table reaches the bottom of Zone A), bottom border `1px solid {colors.hairline-soft}`, tabular numerals right-aligned.
- Highlight the key row or column with a {colors.brand-blue-200} tint or a bold {colors.ink} value, not with shadows.

### Charts
- Pretendard only for every label; axis and legend text at `{typography.caption}` (20px) or larger, direct labels preferred over legends.
- Series use the brand accent order: {colors.brand-blue}, {colors.brand-coral}, {colors.brand-magenta}, {colors.brand-purple}; neutrals in {colors.steel}/{colors.stone}.
- Charts are sized to fill their card or Zone A; no tiny charts floating in white space.
- One idea per chart, titled with the takeaway, with the source in the footer-left.

### Signature Components

**`hero-stat`** — One giant number (`{typography.stat-giant}`, accent or ink) with a label and 2–3 supporting facts beside it; fills Zone A.

**`product-matrix-grid`** — Row of 3–4 equal `card-accent` tiles (one accent per category), 552px tall in Zone A or 264px tall when paired with a second row.

**`comparison-split`** — Two columns (6 + 6): quiet card vs. accent card, with matching rows of attributes.

**`timeline-strip`** — Horizontal line with 4–6 milestones; each milestone is a pill date + `{typography.card-title}` + `{typography.caption}` description.

**`step-row`** — 3–5 numbered cards with numbers in `{typography.stat-lg}`, titles in `{typography.card-title}`, descriptions in `{typography.body}`.

**`testimonial-stat-row`** — Row of 4 stat cells: number in `{typography.stat-lg}`, label in `{typography.caption}` {colors.steel}.

## Logo

- Slides carry **no logo**. Do not draw, place, or reserve space for any logo, wordmark, or placeholder box, and do not mention a missing logo.
- The top-right corner stays clean on every slide. Do not add tag pills, date pills, or any other element there, and do not treat the empty corner as a defect.

## Do's and Don'ts

### Do
- Keep every text and every formula image **inside its box with padding**: at least 12px between the content and its container's edge (pill, chip, card, bar), and at least 16–24px for larger cards. Measure text with the real Pretendard font metrics, and shrink a formula image or enlarge the container instead of letting anything touch or cross a border.
- Use **only Pretendard**, in the weights listed above, for all text.
- Produce **only 1920 × 1080 (16:9)** slides.
- Keep chapter label, title, subtitle and page number at their fixed frame positions on every content slide.
- Follow the Density & Whitespace Rules and Text-First Layout: plain text, few boxes, 2–4 information units, generous gaps.
- Use {colors.primary} (black) as the dominant emphasis color; reserve accent colors for category or emphasis moments.
- Pair {rounded.hero} (32px) accent cards with {rounded.xl} (16px) quiet cards on the same slide for the signature contrast.
- Apply {rounded.full} to every tag, badge, pill and number chip.
- Write one-line titles and one-line subtitles; shorten the copy rather than wrapping or shrinking.
- Keep titles as takeaways ("Revenue doubled after launch"), not labels ("Revenue").

### Don't
- Don't use any font other than Pretendard for text, including monospace for code, and don't use italics in text. Formulas are the only exception and follow the Formulas rules.
- Don't let any text or formula image overflow, touch, or cross the edge of its container.
- Don't create other aspect ratios, and don't add scroll, responsive breakpoints, or web UI chrome (nav bars, buttons, inputs, footers with link columns).
- Don't move or restyle the chapter label, title, subtitle or page number between slides, and don't let a title wrap to a second line.
- Don't put every line of text inside a box, card or pill.
- Don't cram the slide and don't leave an unexplained empty block; don't pad with lorem ipsum or meaningless decoration, and don't shrink text below 18px to fit more in.
- Don't add any logo, wordmark, or logo placeholder to any slide.
- Don't use brand-coral or brand-magenta for body text or large background surfaces.
- Never use gradients anywhere: not on slide backgrounds, cards, pills, badges, charts or diagrams. Use flat solid colors only.
- Never introduce a new component style on a later slide (for example a gradient or colored card on the closing slide). Every slide uses the same plain, text-first visual language.
- Don't apply heavy shadows, glows, or 3D effects.
- Don't mix more than one or two accent colors on a single slide unless the slide encodes four categories.

## Iteration Guide

1. Build every slide on the Slide Frame first (chapter, title, subtitle, footer), then place body modules.
2. Reference tokens directly (`{colors.primary}`, `{typography.title}`, `{rounded.full}`) — do not paraphrase or hard-code values.
3. Headlines step down `cover-title → section-title → title → heading-lg → heading-md → card-title`; default to `{typography.body}` for body and `{typography.body-lg}` for emphasis.
4. After building each slide, check: (a) all text is Pretendard and every formula is a bold LaTeX image, (b) frame elements sit at the fixed coordinates, (c) title and subtitle are one line each, (d) the body follows the Whitespace & Placement Spec (content block 55–75% of the zone, 80–160px below the header rule, 60–115px between side-by-side blocks, ink coverage within limits) with no unexplained empty block, (e) minimum text size is 18px and every text or formula sits inside its container with the required padding, (f) the top-right corner is clean, with no logo, placeholder, tag, or date, (g) the Editability Settings are applied (no snapping, no autofit, native objects), (h) any diagram matches the supplied figure element by element and its arrows follow the Diagrams arrow style.
5. Keep brand accent colors confined to category and emphasis moments. If an accent appears on a generic surface, ask whether it earned that surface.
6. When a slide feels crowded, cut copy or split it into two slides rather than reducing type size; when it feels empty, enlarge the hero visual before adding more boxes.

## Known Gaps

- Dark mode is not supported: all slides are white.
- Animation and slide transitions are not specified; default to none.
- Image treatment (photography, illustration) is not formalized beyond the card radius rules; supplied imagery should be cropped to fill its card without distortion.
