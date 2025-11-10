from __future__ import annotations
from datetime import datetime
from colors import apply as apply_color, resolve as resolve_color, ANSI_RESET as COLORS_ANSI_RESET
from config import (
    ANSI_RESET as CONFIG_ANSI_RESET,
    BORDER_COLORS,
    COMPACT_BACKGROUND_COLORS,
    COMPACT_SYMBOL_COLORS,
    COMPACT_SYMBOLS,
    STATUS_SYMBOLS,
    SUPPRESS_COMPACT_SECOND_HEADER,
    WINDOW_AVAILABLE_SYMBOL as CONFIG_WINDOW_AVAILABLE_SYMBOL,
)
from display_utils import (
    apply_heatmap_header_color,
    display_width,
    extract_row_and_column,
    pad_to_width,
    weekday_short_name,
)
from seatmap_data import SeatMap


class SeatMaps:
    STATUS_SYMBOL = STATUS_SYMBOLS
    WINDOW_AVAILABLE_SYMBOL = CONFIG_WINDOW_AVAILABLE_SYMBOL
    BORDER_COLOR_DEFAULT = BORDER_COLORS['default']
    BORDER_COLOR_BEST = BORDER_COLORS['best']
    BORDER_COLOR_WORST = BORDER_COLORS['worst']
    ANSI_RESET = CONFIG_ANSI_RESET
    COMPACT_BACKGROUND = COMPACT_BACKGROUND_COLORS
    COMPACT_SYMBOLS = COMPACT_SYMBOLS
    COMPACT_SYMBOL_COLORS = COMPACT_SYMBOL_COLORS

    def __init__(self, seatmaps: list[SeatMap] | None = None):
        self.seatmaps = seatmaps or []

    def __iter__(self):
        return iter(self.seatmaps)

    def __len__(self):
        return len(self.seatmaps)

    def add(self, seatmap: SeatMap | None):
        if seatmap is not None:
            self.seatmaps.append(seatmap)

    def render_map(
        self,
        seatmap: SeatMap,
        *,
        highlight: str | None = None,
        style: str = 'ascii',
        thick_border: bool = False,
    ) -> str:
        header = self._format_header(seatmap, style=style)
        render_fn = self._render_ascii_deck if style != 'compact' else self._render_compact_deck
        output = [f"\n{header}"]
        for deck in seatmap.decks:
            rendered = render_fn(deck, highlight=highlight, thick_border=thick_border)
            if rendered:
                output.append(rendered)
        return '\n'.join(output)

    def _render_ascii_deck(self, deck: dict, *, highlight: str | None = None, thick_border: bool = False) -> str:
        rows, column_layout = self._build_seat_grid(deck)
        symbol_width = max(
            [display_width(symbol) for symbol in self.STATUS_SYMBOL.values()] + [display_width(self.WINDOW_AVAILABLE_SYMBOL)]
        )
        seat_column_width = max(symbol_width, 1)
        aisle_column_width = max(1, seat_column_width // 2) + 1

        def format_cell(value: str, width: int) -> str:
            value = value or ''
            pad = max(width - display_width(value), 0)
            return value + (' ' * pad)

        display_columns = [
            (col['position'], col['label'], aisle_column_width if col['is_aisle'] else seat_column_width)
            for col in column_layout
        ]

        header_cells = [format_cell(col_label, width) for _, col_label, width in display_columns]
        header = f"{'':>2} " + ''.join(header_cells).replace('A B   D E F G   J K', ' A B  D E  F G  J K')
        lines = [header]
        for row_name in sorted(rows, key=self._row_sort_key):
            seats_in_row = rows[row_name]
            cells = []
            for pos, _, width in display_columns:
                if pos is None:
                    cells.append(format_cell('', width))
                else:
                    seat_info = seats_in_row.get(pos)
                    symbol = seat_info['symbol'] if seat_info else ' '
                    cells.append(format_cell(symbol, width))
            lines.append(f"{row_name:>2} " + ''.join(cells))

        return self._wrap_with_border(lines, highlight=highlight, thick_border=thick_border)

    def _render_compact_deck(self, deck: dict, *, highlight: str | None = None, thick_border: bool = False) -> str:
        rows, column_layout = self._build_seat_grid(deck)
        if not rows:
            return ''

        header_cells = [' ' if col['is_aisle'] else (col['label'] or ' ') for col in column_layout]
        header = f"{'':>2} " + ''.join(header_cells)
        lines = [header]

        for row_name in sorted(rows, key=self._row_sort_key):
            seats_in_row = rows[row_name]
            row_cells = []
            for col in column_layout:
                if col['is_aisle']:
                    row_cells.append(' ')
                else:
                    row_cells.append(self._compact_seat_cell(seats_in_row.get(col['position'])))
            lines.append(f"{row_name:>2} " + ''.join(row_cells))

        return self._wrap_with_border(lines, highlight=highlight, thick_border=thick_border)

    def _compact_seat_cell(self, seat_info: dict | None) -> str:
        if not seat_info:
            return ' '
        availability = seat_info.get('availability') or 'UNKNOWN'
        is_window = seat_info.get('is_window', False)
        if availability == 'AVAILABLE' and is_window:
            color_key = 'AVAILABLE_WINDOW'
        elif availability == 'AVAILABLE':
            color_key = 'AVAILABLE'
        elif availability == 'OCCUPIED':
            color_key = 'OCCUPIED'
        elif availability == 'BLOCKED':
            color_key = 'BLOCKED'
        else:
            color_key = 'UNKNOWN'
        bg_color = self.COMPACT_BACKGROUND.get(color_key, '')
        symbol = self.COMPACT_SYMBOLS.get(color_key, ' ')
        fg_color = self.COMPACT_SYMBOL_COLORS.get(color_key, '')

        # Resolve tokens (or raw ANSI sequences) to real escape sequences
        bg_seq = resolve_color(bg_color)
        fg_seq = resolve_color(fg_color)

        if not bg_seq:
            # Only foreground (or none)
            if fg_seq:
                return f"{fg_seq}{symbol}{COLORS_ANSI_RESET}"
            return symbol

        # Background present; compose bg + optional fg then symbol, and reset once
        return f"{bg_seq}{fg_seq or ''}{symbol}{COLORS_ANSI_RESET}"

    def _build_seat_grid(self, deck: dict) -> tuple[dict[str, dict[int, dict]], list[dict]]:
        seats = deck.get('seats', [])
        columns_by_position: dict[int, str] = {}
        rows: dict[str, dict[int, dict]] = {}
        for seat in seats:
            coords = seat.get('coordinates', {})
            column_position = coords.get('y')
            if column_position is None:
                continue

            seat_number = seat.get('number', '?')
            row_label, column_label = extract_row_and_column(seat_number)
            columns_by_position.setdefault(column_position, column_label)

            row_bucket = rows.setdefault(row_label, {})
            traveler_pricing = seat.get('travelerPricing', [])
            availability = traveler_pricing[0].get('seatAvailabilityStatus') if traveler_pricing else 'UNKNOWN'
            is_window = 'W' in seat.get('characteristicsCodes', [])
            seat_symbol = self.STATUS_SYMBOL.get(availability, '?')
            if availability == 'AVAILABLE' and is_window:
                seat_symbol = self.WINDOW_AVAILABLE_SYMBOL
            row_bucket[column_position] = {
                'symbol': seat_symbol,
                'availability': availability,
                'is_window': is_window,
            }

        column_layout = self._build_column_layout(columns_by_position)
        return rows, column_layout

    def _build_column_layout(self, columns_by_position: dict[int, str]) -> list[dict]:
        ordered_columns = sorted(columns_by_position)
        layout: list[dict] = []
        last_label = None
        for pos in ordered_columns:
            label = columns_by_position[pos]
            if self._has_aisle_between(last_label, label):
                layout.append({'position': None, 'label': '', 'is_aisle': True})
            layout.append({'position': pos, 'label': label, 'is_aisle': False})
            last_label = label
        return layout

    @staticmethod
    def _has_aisle_between(previous_label: str | None, next_label: str | None) -> bool:
        return (previous_label, next_label) in {('B', 'D'), ('G', 'J')}

    def _format_header(self, seatmap: SeatMap, *, style: str) -> str:
        route = f"{seatmap.origin}{seatmap.destination}".strip()
        flight = f"{seatmap.carrier}{seatmap.number}".strip()
        if style == 'compact':
            date_label = self._format_date_no_year(seatmap.departure_date)
            primary_line, layout = self._build_compact_header_primary_line(date_label, route, flight)
            lines = [primary_line]
            if not SUPPRESS_COMPACT_SECOND_HEADER:
                secondary_line = self._build_compact_header_meta_line(
                    weekday_label=self._weekday_label(seatmap.departure_date),
                    price_label=seatmap.formatted_total_price(rounded=True) or "N/A",
                    aircraft_label=seatmap.aircraft_code or '',
                    layout=layout,
                )
                if secondary_line:
                    lines.append(secondary_line)
            colored_lines = [apply_heatmap_header_color(line) for line in lines]
            return '\n'.join(colored_lines)
        header = f"{seatmap.departure_date} {route} {flight}-{seatmap.aircraft_code} "
        return header

    @staticmethod
    def _format_date_no_year(date_str: str | None) -> str:
        if not date_str:
            return ''
        try:
            date_value = datetime.strptime(date_str, '%Y%m%d')
            return date_value.strftime('%m%d')
        except ValueError:
            return date_str

    @staticmethod
    def _weekday_label(date_str: str | None) -> str:
        if not date_str:
            return ''
        try:
            date_value = datetime.strptime(date_str, '%Y%m%d').date()
        except ValueError:
            return ''
        return weekday_short_name(date_value)

    @staticmethod
    def _build_compact_header_primary_line(
        date_label: str,
        route_label: str,
        flight_label: str,
    ) -> tuple[str, dict[str, int | None]]:
        layout: dict[str, int | None] = {
            'route_start': None,
            'flight_start': None,
            'flight_end': None,
            'line_length': 0,
        }
        builder: list[str] = []
        current_index = 0
        sequence = (
            ('date', date_label),
            ('route', route_label),
            ('flight', flight_label),
        )
        for name, value in sequence:
            if not value:
                continue
            if builder:
                builder.append(' ')
                current_index += 1
            if name == 'route':
                layout['route_start'] = current_index
            if name == 'flight':
                layout['flight_start'] = current_index
            builder.append(value)
            current_index += len(value)
            if name == 'flight':
                layout['flight_end'] = current_index
        line = ''.join(builder)
        layout['line_length'] = len(line)
        return line, layout

    @staticmethod
    def _build_compact_header_meta_line(
        *,
        weekday_label: str,
        price_label: str,
        aircraft_label: str,
        layout: dict[str, int | None],
    ) -> str:
        if not any((weekday_label, price_label, aircraft_label)):
            return ''

        base_length = max(layout.get('line_length', 0) or 0, len(weekday_label))
        if base_length == 0:
            base_length = 1
        characters = [' '] * base_length

        def ensure_capacity(size: int) -> None:
            if size <= len(characters):
                return
            characters.extend([' '] * (size - len(characters)))

        def place_text(text: str, start_index: int) -> None:
            if not text:
                return
            if start_index < 0:
                text = text[-start_index:]
                start_index = 0
            end_index = start_index + len(text)
            ensure_capacity(end_index)
            for offset, char in enumerate(text):
                characters[start_index + offset] = char

        place_text(weekday_label, 0)

        route_start = layout.get('route_start')
        fallback_price_start = len(weekday_label) + 1 if weekday_label else 0
        price_start = route_start if route_start is not None else fallback_price_start
        place_text(price_label, price_start)

        flight_end = layout.get('flight_end')
        if flight_end is None:
            flight_end = max(len(characters), price_start + len(price_label))
        aircraft_start = max(0, flight_end - len(aircraft_label))
        place_text(aircraft_label, aircraft_start)

        rendered = ''.join(characters).rstrip()
        return rendered if rendered.strip() else ''

    def _wrap_with_border(self, lines: list[str], *, highlight: str | None, thick_border: bool) -> str:
        if not lines:
            return ''
        content_width = max((display_width(line) for line in lines), default=0)
        horiz_char = '═' if thick_border else '─'
        vert_char = '║' if thick_border else '│'
        corners = ('╔', '╗', '╚', '╝') if thick_border else ('╭', '╮', '╰', '╯')
        horizontal = horiz_char * (content_width + 2)
        if highlight == 'best':
            border_color = self.BORDER_COLOR_BEST
        elif highlight == 'worst':
            border_color = self.BORDER_COLOR_WORST
        else:
            border_color = self.BORDER_COLOR_DEFAULT

        bordered_lines = [apply_color(border_color, f"{corners[0]}{horizontal}{corners[1]}")]
        left_border = apply_color(border_color, vert_char)
        right_border = apply_color(border_color, vert_char)
        for line in lines:
            padded = self._pad_line(line, content_width)
            bordered_lines.append(f"{left_border} {padded} {right_border}")
        bordered_lines.append(apply_color(border_color, f"{corners[2]}{horizontal}{corners[3]}"))
        return '\n'.join(bordered_lines)

    @staticmethod
    def _pad_line(text: str, width: int) -> str:
        pad = max(width - display_width(text), 0)
        return text + (' ' * pad)

    @staticmethod
    def _row_sort_key(row_name: str):
        return (0, int(row_name)) if row_name.isdigit() else (1, row_name)


def render_text_box(
    lines: list[str],
    *,
    content_width: int,
    content_height: int,
    border_color: str | None = None,
) -> list[str]:
    """Render a text box with padding, returning the list of lines."""
    padded_lines = list(lines)
    while len(padded_lines) < content_height:
        padded_lines.append('')
    padded_lines = [pad_to_width(line, content_width) for line in padded_lines]
    horizontal = '─' * (content_width + 2)
    color = border_color or SeatMaps.BORDER_COLOR_DEFAULT
    box_lines = [apply_color(color, f"╭{horizontal}╮")]
    for line in padded_lines:
        box_lines.append(f"{apply_color(color, '│')} {line} {apply_color(color, '│')}")
    box_lines.append(apply_color(color, f"╰{horizontal}╯"))
    return box_lines


__all__ = ["SeatMaps", "render_text_box"]
