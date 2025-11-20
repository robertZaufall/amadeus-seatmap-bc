import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
import re
import io
import sys

from display_utils import (
    apply_bold_italic,
    apply_emphasis_styles,
    apply_italic_only,
    apply_heatmap_header_color,
    char_display_width,
    display_width,
    pad_to_width,
    pad_to_width_centered,
    resolve_seatmap_style,
    WEEKDAY_SHORT_NAMES,
    weekday_short_name,
)
from seatmap_display import SeatMaps, render_text_box
from seatmap_data import (
    SeatMap,
    build_heatmap_entries,
    build_heatmap_price_stats,
    build_price_entries_all_dates,
    compute_best_price_by_route,
    compute_worst_price_by_route,
    has_best_price_for_route,
    has_worst_price_for_route,
    iter_dates,
    parse_total_price,
)

from config import (
    HEATMAP_COLORS,
    HEATMAP_SYMBOLS,
    SEATMAP_OUTPUT_STYLE,
    SHOW_SEATMAP_PRICE,
    STATIC_LABELS,
    HEATMAP_HEADER_COLOR,
    TRAVEL_WINDOWS,
)
from colors import apply as apply_color, resolve as resolve_color, ANSI_RESET as COLORS_ANSI_RESET
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pillow is optional; PNG export is skipped if unavailable
    Image = None
    ImageDraw = None
    ImageFont = None

travel_windows = TRAVEL_WINDOWS

travel_window_ranges = [
    (
        datetime.fromisoformat(window["start_date"]).date(),
        datetime.fromisoformat(window["end_date"]).date(),
    )
    for window in travel_windows
]

route_travel_windows: dict[str, list[tuple[date, date]]] = defaultdict(list)
for window in travel_windows:
    start = datetime.fromisoformat(window["start_date"]).date()
    end = datetime.fromisoformat(window["end_date"]).date()
    route = f"{window['origin']}->{window['destination']}"
    route_travel_windows[route].append((start, end))

DATA_DIR = Path(__file__).parent / "data" / datetime.now().strftime("%Y%m%d")
SEATMAP_FILE_CANDIDATES: tuple[str, ...] = (
    "seatmap_responses.json",
    "seatmaps_responses.json",
)
RETURN_PRICE_FILE_CANDIDATES: tuple[str, ...] = (
    "pricing_responses_simple.json",
    "prices_responses_simple.json",
)
ONEWAY_PRICE_FILE_CANDIDATES: tuple[str, ...] = (
    "pricing_responses_simple_oneway.json",
    "prices_responses_simple_oneway.json",
    "prices_responses_oneway_simple.json",
)


def _normalize_price_timestamp(value: datetime | str | None) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _resolve_data_file(filename_candidates: tuple[str, ...], base_dir: Path = DATA_DIR) -> Path | None:
    for name in filename_candidates:
        candidate = base_dir / name
        if candidate.exists():
            return candidate
    return None


def _load_json_rows(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]
    if isinstance(payload, list):
        return payload
    return []


def _collect_available_window_seats(decks: list[dict]) -> list[str]:
    available = []
    for deck in decks or []:
        for seat in deck.get("seats", []):
            traveler_pricing = seat.get("travelerPricing", [])
            availability = traveler_pricing[0].get("seatAvailabilityStatus") if traveler_pricing else None
            if availability != "AVAILABLE":
                continue
            if "W" not in seat.get("characteristicsCodes", []):
                continue
            seat_number = seat.get("number")
            if seat_number:
                available.append(seat_number)
    return available


def _load_oneway_price_lookup(base_dir: Path = DATA_DIR) -> dict[tuple[str, str], dict[str, object]]:
    lookup: dict[tuple[str, str], dict[str, object]] = {}
    rows = _load_json_rows(_resolve_data_file(ONEWAY_PRICE_FILE_CANDIDATES, base_dir))
    for entry in rows:
        route = entry.get("outbound_route")
        outbound_date_iso = entry.get("outbound_date")
        price = entry.get("price")
        if not route or not outbound_date_iso or price is None:
            continue
        date_key = str(outbound_date_iso).replace("-", "")
        lookup[(route, date_key)] = {
            "price": str(price),
            "currency": entry.get("currency"),
            "timestamp": _normalize_price_timestamp(entry.get("captured_at")),
        }
    return lookup


def load_seatmaps_from_json(data_dir: Path = DATA_DIR) -> list[SeatMap]:
    """Load seatmaps from the dated JSON fixtures instead of the database."""
    rows = _load_json_rows(_resolve_data_file(SEATMAP_FILE_CANDIDATES, data_dir))
    if not rows:
        return []

    price_lookup = _load_oneway_price_lookup(data_dir)
    seatmaps: list[SeatMap] = []
    for record in rows:
        if not isinstance(record, dict):
            continue
        departure_info = record.get("departure", {}) or {}
        arrival_info = record.get("arrival", {}) or {}
        departure_at = str(departure_info.get("at") or "")
        if "T" in departure_at:
            departure_at = departure_at.split("T", 1)[0]
        if not departure_at:
            continue
        departure_date = departure_at.replace("-", "")
        origin = departure_info.get("iataCode") or ""
        destination = arrival_info.get("iataCode") or ""
        route = f"{origin}->{destination}"
        carrier = record.get("carrierCode") or ""
        number = str(record.get("number") or "")
        aircraft_code = (record.get("aircraft") or {}).get("code") or "N/A"
        decks = record.get("decks") or []
        window_seats = _collect_available_window_seats(decks)
        price_info = price_lookup.get((route, departure_date), {})
        seatmaps.append(
            SeatMap(
                departure_date=departure_date,
                origin=origin,
                destination=destination,
                carrier=carrier,
                number=number,
                aircraft_code=aircraft_code,
                decks=decks,
                window_seats=window_seats,
                price_total=price_info.get("price"),
                price_currency=price_info.get("currency"),
                price_timestamp=price_info.get("timestamp"),
            )
        )
    return seatmaps

seatmap_records = load_seatmaps_from_json()
seatmaps = SeatMaps(seatmap_records)

if not seatmaps:
    print(f"No seat maps available in {DATA_DIR}. Ensure the JSON fixtures are present.")
    exit()

seatmaps_by_date = {
    seatmap_obj.departure_date: seatmap_obj
    for seatmap_obj in seatmap_records
}

best_price_by_route = compute_best_price_by_route(seatmap_records)
worst_price_by_route = compute_worst_price_by_route(seatmap_records)


def build_route_price_metadata(seatmaps_obj: SeatMaps) -> dict[str, dict[str, dict[str, object]]]:
    metadata: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for seatmap_obj in seatmaps_obj:
        route_key = f"{seatmap_obj.origin}->{seatmap_obj.destination}"
        timestamp = _normalize_price_timestamp(getattr(seatmap_obj, "price_timestamp", None))
        metadata[route_key][seatmap_obj.departure_date] = {
            'price': parse_total_price(seatmap_obj.price_total),
            'currency': (seatmap_obj.price_currency or '').upper() or None,
            'timestamp': timestamp,
            'has_window': bool(seatmap_obj.window_seats),
        }
    return metadata


route_price_metadata = build_route_price_metadata(seatmaps)
latest_price_timestamp = max(
    (ts for ts in (seatmap.price_timestamp for seatmap in seatmap_records) if ts),
    default=None,
)


def build_seatmap_context_label(seatmaps_by_date: dict[str, SeatMap]) -> str | None:
    """Return a bold/italic header summarizing routes + months for the seatmap grid."""
    if not seatmaps_by_date:
        return None

    seen_routes: set[tuple[str, str]] = set()
    route_order: list[tuple[str, str]] = []
    month_order: list[tuple[int, int]] = []
    seen_months: set[tuple[int, int]] = set()

    for date_key in sorted(seatmaps_by_date):
        seatmap = seatmaps_by_date[date_key]
        route = (seatmap.origin, seatmap.destination)
        if route not in seen_routes:
            seen_routes.add(route)
            route_order.append(route)
        try:
            parsed_date = datetime.strptime(date_key, '%Y%m%d')
        except ValueError:
            continue
        month_key = (parsed_date.year, parsed_date.month)
        if month_key not in seen_months:
            seen_months.add(month_key)
            month_order.append(month_key)

    route_label = ' / '.join(f"{origin} -> {destination}" for origin, destination in route_order)
    month_label = ' | '.join(datetime(year, month, 1).strftime('%B %Y') for year, month in month_order)
    label_parts = [part for part in (route_label, month_label) if part]
    if not label_parts:
        return None
    return apply_bold_italic(' — '.join(label_parts))


class _TeeWriter:
    """Duplicate stdout writes to a secondary buffer so we can render PNGs later."""
    def __init__(self, *targets):
        self.targets = targets

    def write(self, data: str):
        for target in self.targets:
            target.write(data)

    def flush(self):
        for target in self.targets:
            target.flush()


ANSI_SGR_RE = re.compile(r'\x1b\[([0-9;]*?)m')


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


PNG_FONT_SIZE = 18
EMOJI_FILL_COLORS = {
    "🟩": (32, 180, 120),
    "🟥": (210, 70, 70),
    "🟦": (70, 130, 235),
    "🟨": (230, 200, 70),
    "⬛": (40, 40, 40),
}


def _pick_font(size: int = PNG_FONT_SIZE) -> "ImageFont.FreeTypeFont | ImageFont.ImageFont":
    """Try a few monospace fonts before falling back to default."""
    if ImageFont is None:
        raise RuntimeError("Pillow is required for PNG export; install the 'pillow' package to continue.")
    candidates = [
        "Menlo.ttf",
        "Menlo-Regular.ttf",
        "Consolas.ttf",
        "DejaVuSansMono.ttf",
        "Courier New.ttf",
        "Hack-Regular.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def save_text_block_png(name: str, text: str, output_dir: Path = Path("docs")) -> None:
    """Render ANSI-colored text to a monospaced PNG (dark background)."""
    if not text or not text.strip():
        return
    if Image is None or ImageDraw is None or ImageFont is None:
        print(f"[WARN] Pillow not installed; skipping PNG export for {name}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
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

    lines = text.rstrip('\n').splitlines() or ['']
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

    dest = output_dir / f"{name}.png"
    img.save(dest)


def extract_section(full_text: str, start_marker: str, end_markers: list[str]) -> str:
    """Slice a text block between markers (inclusive of the start marker)."""
    if not start_marker:
        return ''
    start_idx = full_text.find(start_marker)
    if start_idx == -1:
        return ''
    end_positions = [
        pos for pos in (
            full_text.find(marker, start_idx + len(start_marker)) for marker in end_markers
        ) if pos != -1
    ]
    end_idx = min(end_positions) if end_positions else len(full_text)
    return full_text[start_idx:end_idx].strip('\n')


def append_footer(block: str, footer: str | None) -> str:
    if not footer:
        return block
    return f"{block.rstrip()}\n\n{footer}"


def trim_leading_blank_lines(block: str) -> str:
    return block.lstrip('\n')


def window_seat_sort_key(seat_label: str):
    numeric_part = ''.join(filter(str.isdigit, seat_label))
    return int(numeric_part or 0), seat_label

def normalize_block(lines: list[str], width: int, height: int) -> list[str]:
    """Ensure a block of text occupies a consistent rectangle."""
    padded = [pad_to_width(line, width) for line in lines]
    blank_line = ' ' * width
    while len(padded) < height:
        padded.append(blank_line)
    return padded


def is_within_travel_windows(date_key: str) -> bool:
    """Return True if the provided YYYYMMDD date falls inside a travel window."""
    target_date = None
    for fmt in ('%Y%m%d', '%Y-%m-%d'):
        try:
            target_date = datetime.strptime(date_key, fmt).date()
            break
        except ValueError:
            continue
    if target_date is None:
        return False
    for start, end in travel_window_ranges:
        if start <= target_date <= end:
            return True
    return False


def build_placeholder_block(date_key: str, width: int, height: int) -> list[str]:
    """Create a placeholder block for dates without seatmap data."""
    lines: list[str] = []
    show_date = is_within_travel_windows(date_key)
    lines.append(pad_to_width(f"{date_key}" if show_date else '', width))

    blank_line = pad_to_width('', width)
    while len(lines) < height:
        lines.append(blank_line)
    return lines


HEATMAP_SYMBOL_MIN = HEATMAP_SYMBOLS['min']
HEATMAP_SYMBOL_DEFAULT = HEATMAP_SYMBOLS['default']
HEATMAP_SYMBOL_MAX = HEATMAP_SYMBOLS['max']
HEATMAP_COLOR_MIN = HEATMAP_COLORS['min']
HEATMAP_COLOR_DEFAULT = HEATMAP_COLORS['default']
HEATMAP_COLOR_MAX = HEATMAP_COLORS['max']
HEATMAP_CELL_WIDTH = max(
    display_width(HEATMAP_SYMBOL_MIN),
    display_width(HEATMAP_SYMBOL_DEFAULT),
    display_width(HEATMAP_SYMBOL_MAX),
    2,
)
HEATMAP_BLOCK_COLOR_MAP = {
    HEATMAP_COLOR_MIN: 'bg_dark_green',
    HEATMAP_COLOR_DEFAULT: 'bg_yellow',
    HEATMAP_COLOR_MAX: 'bg_dark_red',
}
HEADER_WINDOW_FONT_COLOR = 'fg_white'


def _block_color_for(color: str | None) -> str | None:
    if not color:
        return None
    return HEATMAP_BLOCK_COLOR_MAP.get(color, color)


@dataclass(frozen=True)
class RoundtripAxisEntry:
    date_key: str
    date_obj: date
    price: Decimal | None
    has_window: bool


@dataclass
class RoundtripMatrixData:
    outbound_route: str
    return_route: str
    outbound_axis: list[RoundtripAxisEntry]
    return_axis: list[RoundtripAxisEntry]
    matrix: list[list[Decimal | None]]
    min_value: Decimal | None
    max_value: Decimal | None


def colorize_symbol(symbol: str, color: str | None) -> str:
    if not symbol:
        return ''
    if not color:
        return symbol
    return apply_color(color, symbol)


def heatmap_symbol(
    price: Decimal | None,
    min_price: Decimal | None,
    max_price: Decimal | None,
    *,
    symbol_override: str | None = None,
) -> str:
    if price is None:
        return ''
    if min_price is None or max_price is None or min_price == max_price:
        symbol = HEATMAP_SYMBOL_MIN
        color = HEATMAP_COLOR_MIN
    elif price == min_price:
        symbol = HEATMAP_SYMBOL_MIN
        color = HEATMAP_COLOR_MIN
    elif price == max_price:
        symbol = HEATMAP_SYMBOL_MAX
        color = HEATMAP_COLOR_MAX
    else:
        symbol = HEATMAP_SYMBOL_DEFAULT
        color = HEATMAP_COLOR_DEFAULT

    if symbol_override is not None:
        symbol = symbol_override
        color = _block_color_for(color)
    return colorize_symbol(symbol, color)


def heatmap_color_code(price: Decimal | None, min_price: Decimal | None, max_price: Decimal | None) -> str | None:
    if price is None:
        return None
    if min_price is None or max_price is None or min_price == max_price:
        return HEATMAP_COLOR_MIN
    if price == min_price:
        return HEATMAP_COLOR_MIN
    if price == max_price:
        return HEATMAP_COLOR_MAX
    return HEATMAP_COLOR_DEFAULT


def format_heatmap_calendar(
    route_key: str,
    entries_by_route: dict[str, dict[str, Decimal]],
    stats_by_route: dict[str, tuple[Decimal, Decimal]]
) -> list[str]:
    entries = entries_by_route.get(route_key)
    windows = route_travel_windows.get(route_key)
    if not entries or not windows:
        return []
    min_price, max_price = stats_by_route.get(route_key, (None, None))

    def month_start(value: date) -> date:
        return value.replace(day=1)

    def next_month(value: date) -> date:
        if value.month == 12:
            return value.replace(year=value.year + 1, month=1, day=1)
        return value.replace(month=value.month + 1, day=1)

    def build_month_calendar(
        month_anchor: date,
        window_start: date,
        window_end: date,
    ) -> list[str]:
        month_begin = month_anchor
        month_end = next_month(month_begin) - timedelta(days=1)
        visible_start = max(window_start, month_begin)
        visible_end = min(window_end, month_end)
        if visible_start > visible_end:
            return []

        day_header_plain = ' '.join(pad_to_width(name, HEATMAP_CELL_WIDTH) for name in WEEKDAY_SHORT_NAMES)
        day_header = apply_heatmap_header_color(day_header_plain)
        header_width = display_width(day_header_plain)
        month_year_label = f"{month_begin.strftime('%b')} {month_begin.year}"
        month_year_line = apply_heatmap_header_color(
            pad_to_width_centered(month_year_label, header_width)
        )

        window_lines: list[str] = []
        current = month_begin - timedelta(days=month_begin.weekday())
        month_cutoff = month_end + timedelta(days=(6 - month_end.weekday()))
        while current <= month_cutoff:
            week_boxes: list[str] = []
            week_has_data = False
            for offset in range(7):
                day = current + timedelta(days=offset)
                in_month = month_begin <= day <= month_end
                in_window = visible_start <= day <= visible_end
                day_label = day.strftime('%d') if (in_month and in_window) else ''
                if in_month and in_window:
                    date_key = day.strftime('%Y%m%d')
                    price = entries.get(date_key)
                    if price is not None:
                        week_has_data = True
                        week_boxes.append(
                            heatmap_symbol(
                                price,
                                min_price,
                                max_price,
                                symbol_override=day_label,
                            )
                        )
                    else:
                        week_boxes.append(day_label)
                else:
                    week_boxes.append('')
            if week_has_data:
                formatted_boxes = ' '.join(pad_to_width(token, HEATMAP_CELL_WIDTH) for token in week_boxes)
                window_lines.append(formatted_boxes)
                window_lines.append('')
            current += timedelta(days=7)

        while window_lines and not window_lines[-1].strip():
            window_lines.pop()
        if not window_lines:
            return []
        month_lines: list[str] = [month_year_line, day_header, '']
        month_lines.extend(window_lines)
        return month_lines

    month_blocks: list[list[str]] = []
    for window_start, window_end in windows:
        if window_start > window_end:
            continue
        month_cursor = month_start(window_start)
        while month_cursor <= window_end:
            month_lines = build_month_calendar(month_cursor, window_start, window_end)
            if month_lines:
                content_width = max((display_width(line) for line in month_lines), default=0)
                content_height = len(month_lines)
                month_box = render_text_box(
                    month_lines,
                    content_width=content_width,
                    content_height=content_height,
                    border_color=SeatMaps.BORDER_COLOR_DEFAULT,
                )
                month_blocks.append(month_box)
            month_cursor = next_month(month_cursor)
    if not month_blocks:
        return []

    lines: list[str] = []
    columns = 2
    for idx in range(0, len(month_blocks), columns):
        row_blocks = month_blocks[idx:idx + columns]
        if not row_blocks:
            continue
        block_widths = [
            max((display_width(line) for line in block), default=0)
            for block in row_blocks
        ]
        max_height = max(len(block) for block in row_blocks)
        normalized_blocks: list[list[str]] = []
        for block, width in zip(row_blocks, block_widths):
            padded_lines = [pad_to_width(line, width) for line in block]
            blank_line = ' ' * width
            while len(padded_lines) < max_height:
                padded_lines.append(blank_line)
            normalized_blocks.append(padded_lines)
        for line_idx in range(max_height):
            lines.append('  '.join(block[line_idx] for block in normalized_blocks))
        lines.append('')
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def _resolve_primary_roundtrip_routes() -> tuple[dict, dict, str, str] | None:
    if len(travel_windows) < 2:
        return None
    outbound_window, return_window = travel_windows[:2]
    outbound_route = f"{outbound_window['origin']}->{outbound_window['destination']}"
    return_route = f"{return_window['origin']}->{return_window['destination']}"
    return outbound_window, return_window, outbound_route, return_route


def _build_roundtrip_axes(
    outbound_window: dict,
    return_window: dict,
    *,
    outbound_route: str,
    return_route: str,
    outbound_entries: dict[str, Decimal],
    return_entries: dict[str, Decimal],
    highlight_entries: dict[str, dict[str, Decimal]] | None = None,
) -> tuple[list[RoundtripAxisEntry], list[RoundtripAxisEntry]]:
    def build_axis(
        window: dict,
        route_entries: dict[str, Decimal],
        highlight_lookup: dict[str, Decimal] | None,
    ) -> list[RoundtripAxisEntry]:
        axis: list[RoundtripAxisEntry] = []
        for date_iso in iter_dates(window["start_date"], window["end_date"]):
            date_obj = datetime.fromisoformat(date_iso).date()
            date_key = date_obj.strftime('%Y%m%d')
            has_window = bool(highlight_lookup and date_key in highlight_lookup)
            axis.append(
                RoundtripAxisEntry(
                    date_key=date_key,
                    date_obj=date_obj,
                    price=route_entries.get(date_key),
                    has_window=has_window,
                )
            )
        return axis

    outbound_highlights = highlight_entries.get(outbound_route) if highlight_entries else None
    return_highlights = highlight_entries.get(return_route) if highlight_entries else None
    outbound_axis = build_axis(outbound_window, outbound_entries, outbound_highlights)
    return_axis = build_axis(return_window, return_entries, return_highlights)
    return outbound_axis, return_axis


def _build_axes_from_date_list(
    outbound_dates: list[str],
    return_dates: list[str],
    *,
    outbound_route: str,
    return_route: str,
    highlight_entries: dict[str, dict[str, Decimal]] | None = None,
) -> tuple[list[RoundtripAxisEntry], list[RoundtripAxisEntry]]:
    def build_axis(
        iso_dates: list[str],
        highlight_lookup: dict[str, Decimal] | None,
    ) -> list[RoundtripAxisEntry]:
        axis: list[RoundtripAxisEntry] = []
        for iso_date in iso_dates:
            date_obj = datetime.fromisoformat(iso_date).date()
            date_key = date_obj.strftime('%Y%m%d')
            has_window = bool(highlight_lookup and date_key in highlight_lookup)
            axis.append(
                RoundtripAxisEntry(
                    date_key=date_key,
                    date_obj=date_obj,
                    price=None,
                    has_window=has_window,
                )
            )
        return axis

    outbound_highlights = highlight_entries.get(outbound_route) if highlight_entries else None
    return_highlights = highlight_entries.get(return_route) if highlight_entries else None
    outbound_axis = build_axis(outbound_dates, outbound_highlights)
    return_axis = build_axis(return_dates, return_highlights)
    return outbound_axis, return_axis


def build_roundtrip_matrix_data(
    heatmap_entries: dict[str, dict[str, Decimal]],
    *,
    highlight_entries: dict[str, dict[str, Decimal]] | None = None,
) -> RoundtripMatrixData | None:
    route_info = _resolve_primary_roundtrip_routes()
    if route_info is None or not heatmap_entries:
        return None

    outbound_window, return_window, outbound_route, return_route = route_info
    outbound_entries = heatmap_entries.get(outbound_route, {})
    return_entries = heatmap_entries.get(return_route, {})

    outbound_axis, return_axis = _build_roundtrip_axes(
        outbound_window,
        return_window,
        outbound_route=outbound_route,
        return_route=return_route,
        outbound_entries=outbound_entries,
        return_entries=return_entries,
        highlight_entries=highlight_entries,
    )
    if not outbound_axis or not return_axis:
        return None

    combined_matrix: list[list[Decimal | None]] = []
    combined_values: list[Decimal] = []
    for return_entry in return_axis:
        row: list[Decimal | None] = []
        for outbound_entry in outbound_axis:
            outbound_price = outbound_entry.price
            return_price = return_entry.price
            if outbound_price is not None and return_price is not None:
                combined = outbound_price + return_price
                combined_values.append(combined)
                row.append(combined)
            else:
                row.append(None)
        combined_matrix.append(row)

    min_combined = min(combined_values) if combined_values else None
    max_combined = max(combined_values) if combined_values else None
    return RoundtripMatrixData(
        outbound_route=outbound_route,
        return_route=return_route,
        outbound_axis=outbound_axis,
        return_axis=return_axis,
        matrix=combined_matrix,
        min_value=min_combined,
        max_value=max_combined,
    )


def build_roundtrip_matrix_from_table(
    table_name: str,
    *,
    highlight_entries: dict[str, dict[str, Decimal]] | None = None,
) -> RoundtripMatrixData | None:
    route_info = _resolve_primary_roundtrip_routes()
    if route_info is None:
        return None
    if table_name != "return_prices":
        raise ValueError(f"Unsupported table for roundtrip matrix: {table_name}")

    _, _, outbound_route, return_route = route_info
    rows = _load_json_rows(_resolve_data_file(RETURN_PRICE_FILE_CANDIDATES))
    if not rows:
        return None

    price_lookup: dict[tuple[str, str], Decimal] = {}
    for entry in rows:
        if entry.get("outbound_route") != outbound_route or entry.get("return_route") != return_route:
            continue
        outbound_iso = str(entry.get("outbound_date") or "")
        return_iso = str(entry.get("return_date") or "")
        price_value = entry.get("price")
        if not outbound_iso or not return_iso or price_value is None:
            continue
        try:
            price_decimal = Decimal(str(price_value))
        except (InvalidOperation, ValueError, TypeError):
            continue
        price_lookup[(outbound_iso, return_iso)] = price_decimal

    if not price_lookup:
        return None

    outbound_iso_dates = sorted({key[0] for key in price_lookup})
    return_iso_dates = sorted({key[1] for key in price_lookup})
    if not outbound_iso_dates or not return_iso_dates:
        return None

    def build_continuous_dates(sorted_dates: list[str]) -> list[str]:
        try:
            start_date = datetime.fromisoformat(sorted_dates[0]).date()
            end_date = datetime.fromisoformat(sorted_dates[-1]).date()
        except ValueError:
            return []
        days = (end_date - start_date).days
        return [
            (start_date + timedelta(days=offset)).isoformat()
            for offset in range(days + 1)
        ]

    outbound_range = build_continuous_dates(outbound_iso_dates)
    return_range = build_continuous_dates(return_iso_dates)
    if not outbound_range or not return_range:
        return None

    outbound_axis, return_axis = _build_axes_from_date_list(
        outbound_range,
        return_range,
        outbound_route=outbound_route,
        return_route=return_route,
        highlight_entries=highlight_entries,
    )
    if not outbound_axis or not return_axis:
        return None

    combined_matrix: list[list[Decimal | None]] = []
    combined_values: list[Decimal] = []
    for return_entry in return_axis:
        row: list[Decimal | None] = []
        for outbound_entry in outbound_axis:
            outbound_iso = outbound_entry.date_obj.strftime('%Y-%m-%d')
            return_iso = return_entry.date_obj.strftime('%Y-%m-%d')
            combined_value = price_lookup.get((outbound_iso, return_iso))
            if combined_value is not None:
                combined_values.append(combined_value)
            row.append(combined_value)
        combined_matrix.append(row)

    min_combined = min(combined_values) if combined_values else None
    max_combined = max(combined_values) if combined_values else None
    return RoundtripMatrixData(
        outbound_route=outbound_route,
        return_route=return_route,
        outbound_axis=outbound_axis,
        return_axis=return_axis,
        matrix=combined_matrix,
        min_value=min_combined,
        max_value=max_combined,
    )


def format_roundtrip_price_heatmap(
    heatmap_entries: dict[str, dict[str, Decimal]],
    *,
    title: str | None = None,
    highlight_entries: dict[str, dict[str, Decimal]] | None = None,
    emphasize_highlights: bool = True,
    highlight_emphasis: str = 'config',
    matrix_data: RoundtripMatrixData | None = None,
    show_price_sum_suffix: bool = True,
) -> list[str]:
    """Return a combined outbound/return heatmap covering the first two travel windows."""
    matrix = matrix_data or build_roundtrip_matrix_data(
        heatmap_entries,
        highlight_entries=highlight_entries,
    )
    if matrix is None:
        return []

    outbound_axis = matrix.outbound_axis
    return_axis = matrix.return_axis
    combined_matrix = matrix.matrix
    min_combined = matrix.min_value
    max_combined = matrix.max_value

    def apply_highlight_style(text: str, enabled: bool) -> str:
        if not enabled or not text:
            return text
        if highlight_emphasis == 'italic':
            return apply_italic_only(text)
        if highlight_emphasis == 'bold_italic':
            return apply_bold_italic(text)
        return apply_emphasis_styles(text, enabled=True)

    def style_header_label(text: str, *, available: bool, allow_highlight: bool = True) -> str:
        if not text:
            return text
        if allow_highlight and available:
            return apply_color(HEADER_WINDOW_FONT_COLOR, text)
        return apply_heatmap_header_color(text)

    def format_cell(value: Decimal | None, *, emphasize: bool = False) -> str:
        if value is None:
            return ''
        rounded_value = value.quantize(Decimal('1'), rounding=ROUND_HALF_UP)
        color = heatmap_color_code(value, min_combined, max_combined)
        text = str(rounded_value)
        if color:
            text = apply_color(color, text)
        return apply_highlight_style(text, emphasize)

    rendered_rows: list[list[str]] = []
    should_emphasize = bool(highlight_entries) and emphasize_highlights
    for row_idx, row in enumerate(combined_matrix):
        return_has_window = return_axis[row_idx].has_window
        rendered_row: list[str] = []
        for col_idx, value in enumerate(row):
            outbound_has_window = outbound_axis[col_idx].has_window
            emphasize_cell = should_emphasize and outbound_has_window and return_has_window
            rendered_row.append(format_cell(value, emphasize=emphasize_cell))
        rendered_rows.append(rendered_row)

    def format_date_label(value: date) -> str:
        return value.strftime('%d')

    row_labels = [
        f"{weekday_short_name(entry.date_obj)} {format_date_label(entry.date_obj)}"
        for entry in return_axis
    ]
    column_labels: list[str] = []
    column_weekdays: list[str] = []
    column_months: list[str] = []
    previous_month: str | None = None
    for entry in outbound_axis:
        weekday_label = weekday_short_name(entry.date_obj)
        date_label = format_date_label(entry.date_obj)
        if weekday_label == 'Mo':
            weekday_label = f"|{weekday_label}"
            date_label = f"|{date_label}"
        column_weekdays.append(weekday_label)
        column_labels.append(date_label)
        month_str = entry.date_obj.strftime('%b')
        if month_str != previous_month:
            column_months.append(month_str)
            previous_month = month_str
        else:
            column_months.append('')

    column_split_indices: set[int] = set()
    previous_column_month: int | None = None
    for idx, entry in enumerate(outbound_axis):
        month_value = entry.date_obj.month
        if previous_column_month is not None and month_value != previous_column_month:
            column_split_indices.add(idx)
        previous_column_month = month_value

    row_months: list[str] = []
    previous_row_month: str | None = None
    for entry in return_axis:
        month_str = entry.date_obj.strftime('%b')
        if month_str != previous_row_month:
            row_months.append(month_str)
            previous_row_month = month_str
        else:
            row_months.append('')

    row_split_indices: set[int] = set()
    previous_row_month_value: int | None = None
    for idx, entry in enumerate(return_axis):
        month_value = entry.date_obj.month
        if previous_row_month_value is not None and month_value != previous_row_month_value:
            row_split_indices.add(idx)
        previous_row_month_value = month_value

    cell_width_candidates = [display_width(cell) for row in rendered_rows for cell in row]
    cell_width_candidates.extend(display_width(label) for label in column_labels)
    cell_width_candidates.extend(display_width(label) for label in column_weekdays)
    cell_width_candidates.extend(display_width(label) for label in column_months)
    cell_width = max(cell_width_candidates) if cell_width_candidates else 4

    row_month_header = ''
    row_month_width_candidates = [display_width(row_month_header)]
    row_month_width_candidates.extend(display_width(label) for label in row_months)
    row_month_width = max(row_month_width_candidates) if row_month_width_candidates else 0

    row_label_header = ''
    row_label_width_candidates = [display_width(row_label_header)]
    row_label_width_candidates.extend(display_width(label) for label in row_labels)
    row_label_width = max(row_label_width_candidates) if row_label_width_candidates else 0
    default_title = STATIC_LABELS['roundtrip_title_template'].format(
        outbound_route=matrix.outbound_route,
        return_route=matrix.return_route,
    )
    suffix = "[prices sum two one-way fares]" if show_price_sum_suffix else ""
    title_line = f"{title or default_title} {suffix}".strip()

    separator_color = HEATMAP_HEADER_COLOR

    def paint_separator(text: str) -> str:
        return apply_color(separator_color, text) if separator_color else text

    row_header_separator = paint_separator('│')
    vertical_separator = paint_separator('│')

    def build_separator_line() -> str:
        parts: list[str] = [
            '─' * max(1, row_month_width),
            '─' * max(1, row_label_width),
            '┼',
        ]
        for col_idx in range(len(outbound_axis)):
            if col_idx in column_split_indices:
                parts.append('┼')
            parts.append('─' * max(1, cell_width))
        colored_parts = [paint_separator(part) for part in parts]
        joiner = paint_separator('─')
        return joiner.join(colored_parts)

    content_lines: list[str] = []
    if column_labels:
        left_padding_cells = [
            pad_to_width('', row_month_width),
            pad_to_width('', row_label_width),
            row_header_separator,
        ]
        year_cells = left_padding_cells[:]
        for idx, (axis_entry, label) in enumerate(zip(outbound_axis, column_months)):
            if idx in column_split_indices:
                year_cells.append(vertical_separator)
            styled_label = style_header_label(label, available=axis_entry.has_window, allow_highlight=False)
            year_cells.append(pad_to_width_centered(styled_label, cell_width))
        content_lines.append(' '.join(year_cells))

        weekday_cells = left_padding_cells[:]
        for idx, (axis_entry, label) in enumerate(zip(outbound_axis, column_weekdays)):
            if idx in column_split_indices:
                weekday_cells.append(vertical_separator)
            styled_label = style_header_label(label, available=axis_entry.has_window)
            if label.startswith('|'):
                weekday_cells.append(pad_to_width(styled_label, cell_width))
            else:
                weekday_cells.append(pad_to_width_centered(styled_label, cell_width))
        content_lines.append(' '.join(weekday_cells))

        date_cells = [
            pad_to_width(row_month_header, row_month_width),
            pad_to_width(row_label_header, row_label_width),
            row_header_separator,
        ]
        for idx, (axis_entry, label) in enumerate(zip(outbound_axis, column_labels)):
            if idx in column_split_indices:
                date_cells.append(vertical_separator)
            styled_label = style_header_label(label, available=axis_entry.has_window)
            date_cells.append(pad_to_width_centered(styled_label, cell_width))
        content_lines.append(' '.join(date_cells))
        content_lines.append(build_separator_line())

    horizontal_separator = paint_separator(pad_to_width('─' * max(1, row_month_width), row_month_width))
    horizontal_label_separator = paint_separator(pad_to_width('─' * max(1, row_label_width), row_label_width))
    data_separator_cell = paint_separator(pad_to_width('─' * max(1, cell_width), cell_width))
    cross_separator = paint_separator('┼')

    for row_idx, (axis_entry, month_label, label, row_cells) in enumerate(zip(return_axis, row_months, row_labels, rendered_rows)):
        if row_idx in row_split_indices:
            separator_line_parts = [
                horizontal_separator,
                horizontal_label_separator,
                cross_separator,
            ]
            for col_idx in range(len(outbound_axis)):
                if col_idx in column_split_indices:
                    separator_line_parts.append(cross_separator)
                separator_line_parts.append(data_separator_cell)
            separator_joiner = paint_separator('─')
            content_lines.append(separator_joiner.join(separator_line_parts))
        styled_month = style_header_label(month_label, available=axis_entry.has_window, allow_highlight=False)
        styled_label = style_header_label(label, available=axis_entry.has_window)
        row_line = [
            pad_to_width(styled_month, row_month_width),
            pad_to_width(styled_label, row_label_width),
            row_header_separator,
        ]
        for col_idx, cell in enumerate(row_cells):
            if col_idx in column_split_indices:
                row_line.append(vertical_separator)
            row_line.append(pad_to_width(cell, cell_width))
        content_lines.append(' '.join(row_line))

    content_width = max((display_width(line) for line in content_lines), default=0)
    bordered_lines = render_text_box(
        content_lines,
        content_width=content_width,
        content_height=len(content_lines),
        border_color=SeatMaps.BORDER_COLOR_DEFAULT,
    )
    return ['', title_line, *bordered_lines, '']


def render_availability_boxes(
    route_lines: dict[str, list[str]],
    *,
    route_order: list[str] | None = None,
    heatmap_entries: dict[str, dict[str, Decimal]] | None = None,
    heatmap_stats: dict[str, tuple[Decimal, Decimal]] | None = None,
) -> None:
    if not route_lines:
        return
    if route_order:
        ordered_routes = [route for route in route_order if route in route_lines]
    else:
        ordered_routes = sorted(route_lines)
    box_contents: list[tuple[list[str], list[str] | None]] = []
    for route in ordered_routes:
        entries = route_lines[route] or [STATIC_LABELS['no_window_seats']]
        lines = [route]
        if entries:
            lines.append('')
        lines.extend(entries)
        calendar_box: list[str] | None = None
        if heatmap_entries and heatmap_stats:
            heatmap_lines = format_heatmap_calendar(route, heatmap_entries, heatmap_stats)
            if heatmap_lines:
                calendar_box = heatmap_lines
        box_contents.append((lines, calendar_box))

    if not box_contents:
        return

    content_height = 0
    for base_lines, calendar_box in box_contents:
        total_len = len(base_lines)
        if calendar_box:
            if base_lines and base_lines[-1].strip():
                total_len += 1  # spacer between entries and calendar
            total_len += len(calendar_box)
        content_height = max(content_height, total_len)

    normalized_contents: list[list[str]] = []
    for base_lines, calendar_box in box_contents:
        lines = list(base_lines)
        if calendar_box:
            if lines and lines[-1].strip():
                lines.append('')
            padding_target = content_height - len(calendar_box)
            if len(lines) < padding_target:
                lines.extend([''] * (padding_target - len(lines)))
            lines.extend(calendar_box)
        else:
            if len(lines) < content_height:
                lines.extend([''] * (content_height - len(lines)))
        normalized_contents.append(lines)

    content_width = max(
        (
            (max(display_width(line) for line in lines) if lines else 0)
            for lines in normalized_contents
        ),
        default=0,
    )
    boxes = [
        render_text_box(
            lines,
            content_width=content_width,
            content_height=content_height,
            border_color=SeatMaps.BORDER_COLOR_DEFAULT,
        )
        for lines in normalized_contents
    ]
    if not boxes:
        return
    box_height = len(boxes[0])
    for row_idx in range(box_height):
        row_segments = [box[row_idx] for box in boxes]
        print('  '.join(row_segments))


def print_weekly_layout(
    seatmaps_obj: SeatMaps,
    seatmaps_by_date: dict[str, SeatMap],
    *,
    best_price_by_route: dict[tuple[str, str], Decimal] | None = None,
    worst_price_by_route: dict[tuple[str, str], Decimal] | None = None,
    style: str = 'ascii',
) -> None:
    """Print seatmaps grouped by week, filling missing days with placeholders."""
    if not seatmaps_by_date:
        return

    rendered_blocks: dict[str, list[str]] = {}
    max_width = 0
    max_height = 0
    for date_key, seatmap_obj in seatmaps_by_date.items():
        highlight_state: str | None = None
        if best_price_by_route and has_best_price_for_route(seatmap_obj, best_price_by_route):
            highlight_state = 'best'
        elif worst_price_by_route and has_worst_price_for_route(seatmap_obj, worst_price_by_route):
            highlight_state = 'worst'
        block_lines = seatmaps_obj.render_map(
            seatmap_obj,
            highlight=highlight_state,
            style=style,
            thick_border=bool(seatmap_obj.window_seats),
        ).splitlines()
        while block_lines and not block_lines[0].strip():
            block_lines = block_lines[1:]
        if SHOW_SEATMAP_PRICE:
            price_text = seatmap_obj.formatted_total_price(rounded=True) or "N/A"
            block_lines.append(f"{STATIC_LABELS['price_label']}: {price_text}")
        rendered_blocks[date_key] = block_lines
        if block_lines:
            width = max(display_width(line) for line in block_lines)
            max_width = max(max_width, width)
            max_height = max(max_height, len(block_lines))

    if max_width == 0 or max_height == 0:
        return

    week_row_width = (max_width * 7) + (2 * 6)
    sorted_dates = sorted(datetime.strptime(key, '%Y%m%d') for key in seatmaps_by_date.keys())
    start_date = sorted_dates[0] - timedelta(days=sorted_dates[0].weekday())
    end_date = sorted_dates[-1] + timedelta(days=(6 - sorted_dates[-1].weekday()))
    current = start_date
    placeholder_cache: dict[str, list[str]] = {}
    previous_week_signature: tuple[str, ...] | None = None
    previous_month_key: tuple[int, int] | None = None
    previous_route_label: str | None = None

    while current <= end_date:
        weekly_blocks: list[list[str]] = []
        week_has_data = False
        week_routes: set[str] = set()
        week_data_dates: list[datetime] = []
        for offset in range(7):
            current_date = current + timedelta(days=offset)
            date_key = current_date.strftime('%Y%m%d')
            block_lines = rendered_blocks.get(date_key)
            if block_lines is None:
                block_lines = placeholder_cache.setdefault(
                    date_key,
                    build_placeholder_block(date_key, max_width, max_height)
                )
            else:
                week_has_data = True
                seatmap = seatmaps_by_date.get(date_key)
                if seatmap is not None:
                    week_routes.add(f"{seatmap.origin}->{seatmap.destination}")
                    try:
                        week_data_dates.append(datetime.strptime(date_key, '%Y%m%d'))
                    except ValueError:
                        pass
                block_lines = normalize_block(block_lines, max_width, max_height)
            weekly_blocks.append(block_lines)

        if not week_has_data:
            current += timedelta(days=7)
            continue

        month_reference_date = min(week_data_dates) if week_data_dates else current
        month_key = (month_reference_date.year, month_reference_date.month)
        if month_key != previous_month_key:
            route_label = ' / '.join(sorted(week_routes)) if week_routes else ''
            month_label = datetime(month_reference_date.year, month_reference_date.month, 1).strftime('%B %Y')
            if route_label and route_label != previous_route_label:
                boxed_route = render_text_box(
                    [apply_bold_italic(route_label)],
                    content_width=display_width(route_label),
                    content_height=1,
                    border_color=SeatMaps.BORDER_COLOR_DEFAULT,
                )
                for line in boxed_route:
                    print(line)
                print()
                previous_route_label = route_label
            styled_month = apply_bold_italic(month_label)
            underline_width = max(week_row_width, display_width(styled_month))
            underline = apply_color(HEATMAP_HEADER_COLOR, '─' * underline_width) if underline_width > 0 else ''
            print(styled_month)
            if underline:
                print(underline)
            print()
            previous_month_key = month_key

        week_signature = tuple(sorted(week_routes))
        if previous_week_signature and week_signature != previous_week_signature:
            print()

        for line_idx in range(max_height):
            print('  '.join(block[line_idx] for block in weekly_blocks))
        print()

        previous_week_signature = week_signature
        current += timedelta(days=7)


original_stdout = sys.stdout
output_buffer = io.StringIO()
sys.stdout = _TeeWriter(original_stdout, output_buffer)

selected_seatmap_styles = resolve_seatmap_style(SEATMAP_OUTPUT_STYLE)

print("\n\n")

seatmap_context_label = build_seatmap_context_label(seatmaps_by_date)
section_slices: dict[str, str] = {}
last_slice_end = 0
for idx, style in enumerate(selected_seatmap_styles):
    if seatmaps_by_date:
        if style == 'compact':
            print(STATIC_LABELS['compact_seatmap_heading'])
        elif len(selected_seatmap_styles) > 1:
            normal_heading = STATIC_LABELS.get('normal_seatmap_heading')
            if normal_heading:
                print(normal_heading)
    print_weekly_layout(
        seatmaps,
        seatmaps_by_date,
        best_price_by_route=best_price_by_route,
        worst_price_by_route=worst_price_by_route,
        style=style,
    )
    # Capture the newly appended chunk for this style to avoid double-rendering in PNGs.
    current_text = output_buffer.getvalue()
    section_slices[style] = current_text[last_slice_end:]
    last_slice_end = len(current_text)
    if idx < len(selected_seatmap_styles) - 1:
        print()

if seatmaps_by_date:
    print(STATIC_LABELS['availability_heading'])
    heatmap_entries = build_heatmap_entries(seatmaps_by_date, travel_windows)
    heatmap_stats = build_heatmap_price_stats(heatmap_entries)
    all_price_entries = build_price_entries_all_dates(seatmaps_by_date, travel_windows)
    availability_by_route: dict[str, list[str]] = {}
    route_first_date: dict[str, str] = {}
    for date_key in sorted(seatmaps_by_date):
        seatmap = seatmaps_by_date[date_key]
        route_key = f"{seatmap.origin}->{seatmap.destination}"
        route_first_date.setdefault(route_key, date_key)
        seats = seatmap.window_seats
        if not seats:
            availability_by_route.setdefault(route_key, [])
            continue
        sorted_seats = ', '.join(sorted(seats, key=window_seat_sort_key))
        formatted_date = datetime.strptime(date_key, '%Y%m%d').strftime('%Y-%m-%d')
        price_text = seatmap.formatted_total_price(rounded=True) or "N/A"
        price_decimal = heatmap_entries.get(route_key, {}).get(date_key)
        min_price, max_price = heatmap_stats.get(route_key, (None, None))
        colored_date = apply_heatmap_header_color(formatted_date)
        price_color = heatmap_color_code(price_decimal, min_price, max_price)
        colored_price = apply_color(price_color, price_text) if price_color else price_text
        availability_by_route.setdefault(route_key, []).append(
            f"{colored_date} {colored_price}: {sorted_seats}"
        )

    if availability_by_route:
        ordered_routes = sorted(route_first_date, key=route_first_date.get)
        render_availability_boxes(
            availability_by_route,
            route_order=ordered_routes,
            heatmap_entries=heatmap_entries,
            heatmap_stats=heatmap_stats,
        )

    window_matrix = build_roundtrip_matrix_data(heatmap_entries)
    window_roundtrip_heatmap = format_roundtrip_price_heatmap(
        heatmap_entries,
        title=STATIC_LABELS['roundtrip_window_title'],
        matrix_data=window_matrix,
    ) if window_matrix else []
    if window_roundtrip_heatmap:
        print()
        for line in window_roundtrip_heatmap:
            print(line)

    all_price_matrix = build_roundtrip_matrix_data(
        all_price_entries,
        highlight_entries=heatmap_entries,
    )
    if all_price_matrix:
        all_price_roundtrip_heatmap = format_roundtrip_price_heatmap(
            all_price_entries,
            title=STATIC_LABELS['roundtrip_all_title'],
            highlight_entries=heatmap_entries,
            emphasize_highlights=True,
            highlight_emphasis='bold_italic',
            matrix_data=all_price_matrix,
        )
    else:
        all_price_roundtrip_heatmap = []
    if all_price_roundtrip_heatmap:
        for line in all_price_roundtrip_heatmap:
            print(line)

    return_price_matrix = build_roundtrip_matrix_from_table(
        "return_prices",
        highlight_entries=heatmap_entries,
    )
    return_price_heatmap = (
        format_roundtrip_price_heatmap(
            heatmap_entries,
            title=STATIC_LABELS['roundtrip_return_prices_title'],
            highlight_entries=heatmap_entries,
            matrix_data=return_price_matrix,
            show_price_sum_suffix=False,
        )
        if return_price_matrix
        else []
    )
    if return_price_heatmap:
        for line in return_price_heatmap:
            print(line)

print("\n")

# Restore stdout and emit PNG snapshots for each major section
sys.stdout = original_stdout
full_output = output_buffer.getvalue()

def _clean_marker(value: str | None) -> str:
    return (value or '').strip()

compact_marker = _clean_marker(STATIC_LABELS.get('compact_seatmap_heading'))
normal_marker = _clean_marker(STATIC_LABELS.get('normal_seatmap_heading'))
availability_marker = _clean_marker(STATIC_LABELS.get('availability_heading'))
timestamp_label = (
    f"Data timestamp: {latest_price_timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
    if latest_price_timestamp
    else "Data timestamp: N/A"
)

compact_section = section_slices.get('compact') or extract_section(
    full_output,
    compact_marker,
    [marker for marker in (normal_marker, availability_marker) if marker],
)
if compact_section:
    save_text_block_png("seatmaps_compact", append_footer(trim_leading_blank_lines(compact_section), timestamp_label))

normal_section = section_slices.get('ascii') or extract_section(
    full_output,
    normal_marker,
    [marker for marker in (availability_marker, "Round-trip price heatmap") if marker],
)
if normal_section:
    save_text_block_png("seatmaps", append_footer(trim_leading_blank_lines(normal_section), timestamp_label))

availability_section = extract_section(
    full_output,
    availability_marker,
    ["Round-trip price heatmap"],
)
if availability_section:
    save_text_block_png("window_seats", append_footer(trim_leading_blank_lines(availability_section), timestamp_label))

heatmaps_section = extract_section(full_output, "Round-trip price heatmap", [])
if heatmaps_section:
    save_text_block_png("price_heatmaps", append_footer(trim_leading_blank_lines(heatmaps_section), timestamp_label))
