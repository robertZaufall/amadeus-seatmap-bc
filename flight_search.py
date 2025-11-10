from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP

from display_utils import (
    apply_bold_italic,
    apply_emphasis_styles,
    apply_italic_only,
    apply_heatmap_header_color,
    display_width,
    pad_to_width,
    pad_to_width_centered,
    resolve_seatmap_style,
    WEEKDAY_SHORT_NAMES,
    weekday_short_name,
)
from seatmap_display import SeatMaps, render_text_box
from seatmap_data import (
    CombinationPriceRecord,
    SeatMap,
    build_heatmap_entries,
    build_heatmap_price_stats,
    build_price_entries_all_dates,
    compute_best_price_by_route,
    compute_worst_price_by_route,
    has_best_price_for_route,
    has_worst_price_for_route,
    iter_dates,
    load_seatmaps,
    parse_total_price,
    sync_combination_price_records,
)
from dotenv import load_dotenv

from config import (
    ENVIRONMENT,
    HEATMAP_COLORS,
    HEATMAP_SYMBOLS,
    FLIGHT_SEARCH_FILTERS,
    SEATMAP_OUTPUT_STYLE,
    SHOW_SEATMAP_PRICE,
    STATIC_LABELS,
    TRAVEL_WINDOWS,
)
from colors import apply as apply_color, resolve as resolve_color, ANSI_RESET as COLORS_ANSI_RESET

load_dotenv()

environment = ENVIRONMENT
travel_windows = TRAVEL_WINDOWS
flight_search_filters = FLIGHT_SEARCH_FILTERS

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

seatmaps = SeatMaps(load_seatmaps(
    environment=environment,
    travel_windows=travel_windows,
    flight_search_filters=flight_search_filters,
))

if not seatmaps:
    print("No seat maps available")
    exit()

seatmaps_by_date = {
    seatmap_obj.departure_date: seatmap_obj
    for seatmap_obj in seatmaps
}

best_price_by_route = compute_best_price_by_route(seatmaps)
worst_price_by_route = compute_worst_price_by_route(seatmaps)


def _normalize_price_timestamp(value):
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


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


HEATMAP_SYMBOL_FALLBACK = colorize_symbol(HEATMAP_SYMBOL_DEFAULT, HEATMAP_COLOR_DEFAULT)


def heatmap_symbol(price: Decimal | None, min_price: Decimal | None, max_price: Decimal | None) -> str:
    if price is None:
        return ''
    if min_price is None or max_price is None or min_price == max_price:
        return colorize_symbol(HEATMAP_SYMBOL_MIN, HEATMAP_COLOR_MIN)
    if price == min_price:
        return colorize_symbol(HEATMAP_SYMBOL_MIN, HEATMAP_COLOR_MIN)
    if price == max_price:
        return colorize_symbol(HEATMAP_SYMBOL_MAX, HEATMAP_COLOR_MAX)
    return colorize_symbol(HEATMAP_SYMBOL_DEFAULT, HEATMAP_COLOR_DEFAULT)


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

    lines: list[str] = ['', '']
    for window_start, window_end in windows:
        if window_start > window_end:
            continue
        window_lines: list[str] = []
        day_header_plain = ' '.join(pad_to_width(name, HEATMAP_CELL_WIDTH) for name in WEEKDAY_SHORT_NAMES)
        day_header = apply_heatmap_header_color(day_header_plain)
        current = window_start - timedelta(days=window_start.weekday())
        window_cutoff = window_end + timedelta(days=(6 - window_end.weekday()))
        while current <= window_cutoff:
            week_boxes: list[str] = []
            week_labels: list[str] = []
            week_has_data = False
            for offset in range(7):
                day = current + timedelta(days=offset)
                in_window = window_start <= day <= window_end
                if in_window:
                    date_key = day.strftime('%Y%m%d')
                    price = entries.get(date_key)
                    day_label = day.strftime('%d')
                    if price is not None:
                        week_has_data = True
                        week_boxes.append(heatmap_symbol(price, min_price, max_price))
                        week_labels.append(day_label)
                    else:
                        week_boxes.append('')
                        week_labels.append(day_label)
                else:
                    week_boxes.append('')
                    week_labels.append('')
            if week_has_data:
                formatted_boxes = ' '.join(pad_to_width(token, HEATMAP_CELL_WIDTH) for token in week_boxes)
                formatted_labels = ' '.join(pad_to_width(label, HEATMAP_CELL_WIDTH) for label in week_labels)
                window_lines.append(formatted_boxes)
                window_lines.append(formatted_labels)
            current += timedelta(days=7)
        if window_lines:
            lines.append(day_header)
            lines.extend(window_lines)
            lines.append('')
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def build_roundtrip_matrix_data(
    heatmap_entries: dict[str, dict[str, Decimal]],
    *,
    highlight_entries: dict[str, dict[str, Decimal]] | None = None,
) -> RoundtripMatrixData | None:
    if len(travel_windows) < 2 or not heatmap_entries:
        return None

    outbound_window, return_window = travel_windows[:2]
    outbound_route = f"{outbound_window['origin']}->{outbound_window['destination']}"
    return_route = f"{return_window['origin']}->{return_window['destination']}"
    outbound_entries = heatmap_entries.get(outbound_route, {})
    return_entries = heatmap_entries.get(return_route, {})

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

    outbound_axis = build_axis(
        outbound_window,
        outbound_entries,
        highlight_entries.get(outbound_route) if highlight_entries else None,
    )
    return_axis = build_axis(
        return_window,
        return_entries,
        highlight_entries.get(return_route) if highlight_entries else None,
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


def format_roundtrip_price_heatmap(
    heatmap_entries: dict[str, dict[str, Decimal]],
    *,
    title: str | None = None,
    highlight_entries: dict[str, dict[str, Decimal]] | None = None,
    emphasize_highlights: bool = True,
    highlight_emphasis: str = 'config',
    matrix_data: RoundtripMatrixData | None = None,
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
        return value.strftime('%m%d')

    row_labels = [
        f"{weekday_short_name(entry.date_obj)} {format_date_label(entry.date_obj)}"
        for entry in return_axis
    ]
    column_labels = [format_date_label(entry.date_obj) for entry in outbound_axis]
    column_weekdays = [weekday_short_name(entry.date_obj) for entry in outbound_axis]
    column_years: list[str] = []
    previous_year: str | None = None
    for entry in outbound_axis:
        year_str = entry.date_obj.strftime('%Y')
        if year_str != previous_year:
            column_years.append(year_str)
            previous_year = year_str
        else:
            column_years.append('')

    row_years: list[str] = []
    previous_row_year: str | None = None
    for entry in return_axis:
        year_str = entry.date_obj.strftime('%Y')
        if year_str != previous_row_year:
            row_years.append(year_str)
            previous_row_year = year_str
        else:
            row_years.append('')

    cell_width_candidates = [display_width(cell) for row in rendered_rows for cell in row]
    cell_width_candidates.extend(display_width(label) for label in column_labels)
    cell_width_candidates.extend(display_width(label) for label in column_weekdays)
    cell_width_candidates.extend(display_width(label) for label in column_years)
    cell_width = max(cell_width_candidates) if cell_width_candidates else 4

    row_year_header = ''
    row_year_width_candidates = [display_width(row_year_header)]
    row_year_width_candidates.extend(display_width(label) for label in row_years)
    row_year_width = max(row_year_width_candidates) if row_year_width_candidates else 0

    row_label_header = ''
    row_label_width_candidates = [display_width(row_label_header)]
    row_label_width_candidates.extend(display_width(label) for label in row_labels)
    row_label_width = max(row_label_width_candidates) if row_label_width_candidates else 0
    default_title = STATIC_LABELS['roundtrip_title_template'].format(
        outbound_route=matrix.outbound_route,
        return_route=matrix.return_route,
    )
    suffix = "[prices sum two one-way fares] "
    title_line = f"{title or default_title} {suffix}"
    content_lines: list[str] = []
    if column_labels:
        left_padding_cells = [
            pad_to_width('', row_year_width),
            pad_to_width('', row_label_width),
        ]
        year_cells = left_padding_cells[:]
        year_cells.extend(
            pad_to_width_centered(apply_heatmap_header_color(label), cell_width) for label in column_years
        )
        content_lines.append(' '.join(year_cells))

        weekday_cells = left_padding_cells[:]
        weekday_cells.extend(
            pad_to_width_centered(apply_heatmap_header_color(label), cell_width) for label in column_weekdays
        )
        content_lines.append(' '.join(weekday_cells))

        date_cells = [
            pad_to_width(row_year_header, row_year_width),
            pad_to_width(row_label_header, row_label_width),
        ]
        date_cells.extend(
            pad_to_width(apply_heatmap_header_color(label), cell_width) for label in column_labels
        )
        content_lines.append(' '.join(date_cells))

    for year_label, label, row_cells in zip(row_years, row_labels, rendered_rows):
        row_line = [
            pad_to_width(apply_heatmap_header_color(year_label), row_year_width),
            pad_to_width(apply_heatmap_header_color(label), row_label_width),
        ]
        row_line.extend(pad_to_width(cell, cell_width) for cell in row_cells)
        content_lines.append(' '.join(row_line))

    content_width = max((display_width(line) for line in content_lines), default=0)
    bordered_lines = render_text_box(
        content_lines,
        content_width=content_width,
        content_height=len(content_lines),
        border_color=SeatMaps.BORDER_COLOR_DEFAULT,
    )
    return ['', title_line, *bordered_lines, '']


def _resolve_combination_timestamp(
    outbound_info: dict[str, object] | None,
    return_info: dict[str, object] | None,
) -> datetime | None:
    timestamps: list[datetime] = []
    for info in (outbound_info, return_info):
        if not info:
            continue
        ts = _normalize_price_timestamp(info.get('timestamp'))
        if ts:
            timestamps.append(ts)
    if not timestamps:
        return None
    return min(timestamps)


def _resolve_combination_currency(
    outbound_info: dict[str, object] | None,
    return_info: dict[str, object] | None,
) -> str | None:
    currencies = [
        (info.get('currency') if info else None)
        for info in (outbound_info, return_info)
        if info and info.get('currency')
    ]
    if not currencies:
        return None
    first = currencies[0]
    if all(currency == first for currency in currencies):
        return first
    return None


def build_combination_price_records(
    matrix: RoundtripMatrixData,
    route_price_metadata: dict[str, dict[str, dict[str, object]]],
) -> tuple[list[CombinationPriceRecord], list[tuple[str, str, str, str]]]:
    outbound_meta = route_price_metadata.get(matrix.outbound_route, {})
    return_meta = route_price_metadata.get(matrix.return_route, {})
    records: list[CombinationPriceRecord] = []
    missing_keys: list[tuple[str, str, str, str]] = []

    for row_idx, row in enumerate(matrix.matrix):
        return_axis_entry = matrix.return_axis[row_idx]
        return_info = return_meta.get(return_axis_entry.date_key)
        for col_idx, value in enumerate(row):
            outbound_axis_entry = matrix.outbound_axis[col_idx]
            key = (
                matrix.outbound_route,
                matrix.return_route,
                outbound_axis_entry.date_key,
                return_axis_entry.date_key,
            )
            if value is None:
                missing_keys.append(key)
                continue
            outbound_info = outbound_meta.get(outbound_axis_entry.date_key)
            timestamp = _resolve_combination_timestamp(outbound_info, return_info) or datetime.utcnow()
            currency = _resolve_combination_currency(outbound_info, return_info)
            outbound_has_window = outbound_info.get('has_window') if outbound_info else None
            return_has_window = return_info.get('has_window') if return_info else None
            records.append(
                CombinationPriceRecord(
                    outbound_route=matrix.outbound_route,
                    return_route=matrix.return_route,
                    outbound_date=outbound_axis_entry.date_key,
                    return_date=return_axis_entry.date_key,
                    price=value,
                    currency=currency,
                    captured_at=timestamp,
                    outbound_window_available=outbound_has_window,
                    return_window_available=return_has_window,
                )
            )
    return records, missing_keys


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
    box_contents: list[list[str]] = []
    for route in ordered_routes:
        entries = route_lines[route] or [STATIC_LABELS['no_window_seats']]
        lines = [route]
        if entries:
            lines.append('')
        lines.extend(entries)
        if heatmap_entries and heatmap_stats:
            heatmap_lines = format_heatmap_calendar(route, heatmap_entries, heatmap_stats)
            if heatmap_lines:
                lines.extend(heatmap_lines)
        box_contents.append(lines)

    content_width = max(
        ((max(display_width(line) for line in lines) if lines else 0) for lines in box_contents),
        default=0,
    )
    content_height = max((len(lines) for lines in box_contents), default=0)
    boxes = [
        render_text_box(
            lines,
            content_width=content_width,
            content_height=content_height,
            border_color=SeatMaps.BORDER_COLOR_DEFAULT,
        )
        for lines in box_contents
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

    sorted_dates = sorted(datetime.strptime(key, '%Y%m%d') for key in seatmaps_by_date.keys())
    start_date = sorted_dates[0] - timedelta(days=sorted_dates[0].weekday())
    end_date = sorted_dates[-1] + timedelta(days=(6 - sorted_dates[-1].weekday()))
    current = start_date
    placeholder_cache: dict[str, list[str]] = {}
    previous_week_signature: tuple[str, ...] | None = None

    while current <= end_date:
        weekly_blocks: list[list[str]] = []
        week_has_data = False
        week_routes: set[str] = set()
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
                block_lines = normalize_block(block_lines, max_width, max_height)
            weekly_blocks.append(block_lines)

        if not week_has_data:
            current += timedelta(days=7)
            continue

        week_signature = tuple(sorted(week_routes))
        if previous_week_signature and week_signature != previous_week_signature:
            print()

        for line_idx in range(max_height):
            print('  '.join(block[line_idx] for block in weekly_blocks))
        print()

        previous_week_signature = week_signature
        current += timedelta(days=7)

selected_seatmap_style = resolve_seatmap_style(SEATMAP_OUTPUT_STYLE)

print("\n\n")
if seatmaps_by_date and selected_seatmap_style == 'compact':
    print(STATIC_LABELS['compact_seatmap_heading'])
print_weekly_layout(
    seatmaps,
    seatmaps_by_date,
    best_price_by_route=best_price_by_route,
    worst_price_by_route=worst_price_by_route,
    style=selected_seatmap_style,
)

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
        symbol_prefix = heatmap_symbol(price_decimal, min_price, max_price) or HEATMAP_SYMBOL_FALLBACK
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
        records, missing_keys = build_combination_price_records(all_price_matrix, route_price_metadata)
        sync_combination_price_records(records, missing_keys)
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

print("\n")
