from __future__ import annotations

import re
from pathlib import Path

from display_utils import char_display_width

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pillow is optional; PNG export is skipped if unavailable
    Image = None
    ImageDraw = None
    ImageFont = None


ANSI_SGR_RE = re.compile(r'\x1b\[([0-9;]*?)m')
PNG_FONT_SIZE = 18
EMOJI_FILL_COLORS = {
    "🟩": (32, 180, 120),
    "🟥": (210, 70, 70),
    "🟦": (70, 130, 235),
    "🟨": (230, 200, 70),
    "⬛": (40, 40, 40),
}


def _ansi_palette() -> dict[int, tuple[int, int, int]]:
    """Basic 16-color ANSI palette (xterm-ish)."""
    return {
        30: (0, 0, 0),          # black
        31: (205, 49, 49),      # red
        32: (13, 188, 121),     # green
        33: (229, 229, 16),     # yellow
        34: (65, 140, 220),     # blue (brightened a bit)
        35: (188, 63, 188),     # magenta
        36: (17, 168, 205),     # cyan
        37: (229, 229, 229),    # white
        90: (160, 160, 160),    # bright black (grey) - lifted for contrast
        91: (241, 76, 76),
        92: (35, 209, 139),
        93: (245, 245, 67),
        94: (59, 142, 234),
        95: (214, 112, 214),
        96: (41, 184, 219),
        97: (255, 255, 255),
        40: (0, 0, 0),          # bg black
        41: (205, 49, 49),
        42: (13, 188, 121),
        43: (229, 229, 16),
        44: (36, 114, 200),
        45: (188, 63, 188),
        46: (17, 168, 205),
        47: (229, 229, 229),
        100: (70, 70, 70),      # bg bright black (lightened)
        101: (241, 76, 76),
        102: (35, 209, 139),
        103: (245, 245, 67),
        104: (59, 142, 234),
        105: (214, 112, 214),
        106: (41, 184, 219),
        107: (255, 255, 255),
    }


def _brighten(color: tuple[int, int, int], factor: float = 1.15) -> tuple[int, int, int]:
    return tuple(min(int(channel * factor), 255) for channel in color)


def _pick_font(size: int = PNG_FONT_SIZE) -> "ImageFont.FreeTypeFont | ImageFont.ImageFont":
    """Try a few monospace fonts before falling back to default."""
    if ImageFont is None:
        raise RuntimeError("Pillow is required for PNG export; install the 'pillow' package to continue.")
    candidates = [
        "DejaVuSansMono.ttf",
        "Menlo.ttf",
        "Menlo-Regular.ttf",
        "Consolas.ttf",
        "Courier New.ttf",
        "Hack-Regular.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def save_text_block_png(
    name: str,
    text: str,
    output_dir: Path = Path("docs"),
    *,
    occupied_replacement: str = "XX",
    output_path: Path | None = None,
) -> None:
    """Render ANSI-colored text to a monospaced PNG (dark background)."""
    if not text or not text.strip():
        return
    if Image is None or ImageDraw is None or ImageFont is None:
        print(f"[WARN] Pillow not installed; skipping PNG export for {name}")
        return

    # Fallback glyphs that render reliably in common monospace fonts (PNG only).
    # Normalize occupied-seat glyphs for PNG export; keep terminal output unchanged.
    occupied_safe = occupied_replacement or "XX"
    safe_text = text.translate(str.maketrans({"❌": occupied_safe, "🔺": occupied_safe, "✘": occupied_safe, "✖": occupied_safe}))

    target_dir = output_path.parent if output_path else output_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    font = _pick_font()
    palette = _ansi_palette()
    base_bg = (18, 18, 18)
    base_fg = (235, 235, 235)
    margin = 12
    padding_x = 1
    line_spacing = 4

    def iter_cells(line: str):
        fg = base_fg
        bg = base_bg
        bold = False
        idx = 0
        length = len(line)
        while idx < length:
            ch = line[idx]
            if ch == '\x1b':
                match = ANSI_SGR_RE.match(line, idx)
                if match:
                    codes = [code for code in match.group(1).split(';') if code != '']
                    if not codes:
                        codes = ['0']
                    for code in codes:
                        if code == '0':
                            fg, bg, bold = base_fg, base_bg, False
                        elif code == '1':
                            bold = True
                        elif code == '3':  # italic (ignored for layout)
                            continue
                        else:
                            num = int(code) if code.isdigit() else None
                            if num is not None:
                                if 30 <= num <= 37 or 90 <= num <= 97:
                                    fg = palette.get(num, base_fg)
                                elif 40 <= num <= 47 or 100 <= num <= 107:
                                    bg = palette.get(num, base_bg)
                    idx = match.end()
                    continue
            width = max(1, char_display_width(ch))
            fg_effective = fg if not bold else _brighten(fg)
            # High-contrast digits when background is set (calendar cells)
            if bg != base_bg and ch.isdigit():
                fg_effective = (0, 0, 0)
            yield ch, width, fg_effective, bg
            idx += 1

    lines = safe_text.rstrip('\n').splitlines() or ['']
    line_cell_widths = []
    parsed_lines = []
    for raw_line in lines:
        cells = list(iter_cells(raw_line))
        parsed_lines.append(cells)
        line_cell_widths.append(sum(cell[1] for cell in cells))
    max_cells = max(line_cell_widths or [1])

    glyph_bbox = font.getbbox('M')
    cell_width = max((glyph_bbox[2] - glyph_bbox[0]) + padding_x, 8)
    ascent, descent = font.getmetrics()
    line_height = ascent + descent + line_spacing

    img_width = margin * 2 + cell_width * max_cells
    img_height = margin * 2 + line_height * len(lines)
    img = Image.new("RGB", (img_width, img_height), color=base_bg)
    draw = ImageDraw.Draw(img)

    y = margin
    for cells in parsed_lines:
        x = margin
        for ch, cell_w, fg, bg in cells:
            cell_right = x + cell_width * cell_w
            cell_bottom = y + line_height - line_spacing // 3
            if bg != base_bg:
                draw.rectangle(
                    [x, y - line_spacing // 3, cell_right, cell_bottom],
                    fill=bg,
                )
            if ch in EMOJI_FILL_COLORS:
                # Draw solid block to approximate emoji color for fonts without emoji glyphs.
                draw.rectangle(
                    [x + 1, y + 1, cell_right - 1, cell_bottom - 2],
                    fill=EMOJI_FILL_COLORS[ch],
                )
            else:
                draw.text(
                    (x, y),
                    ch,
                    fill=fg,
                    font=font,
                    stroke_width=1,
                    stroke_fill=fg,
                )
            x += cell_width * cell_w
        y += line_height

    dest = output_path if output_path else target_dir / f"{name}.png"
    img.save(dest)
