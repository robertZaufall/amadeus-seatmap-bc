from __future__ import annotations
from datetime import datetime
from colors import apply as apply_color, resolve as resolve_color, ANSI_RESET as COLORS_ANSI_RESET
from config import (
    ANSI_RESET as CONFIG_ANSI_RESET,
    BORDER_COLORS,
    COMPACT_BACKGROUND_COLORS,
    COMPACT_SYMBOL_COLORS,
    COMPACT_SYMBOLS,
    HIGHLIGHT_AVAILABLE_SYMBOL,
    HIGHLIGHT_CHARACTERISTIC_CODES,
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
    HIGHLIGHT_CHARACTERISTIC_CODES = HIGHLIGHT_CHARACTERISTIC_CODES
    HIGHLIGHT_AVAILABLE_SYMBOL = HIGHLIGHT_AVAILABLE_SYMBOL

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
        show_header: bool = True,
    ) -> str:
        render_fn = self._render_ascii_deck if style != 'compact' else self._render_compact_deck
        rendered_decks: list[str] = []
        for deck in self._sorted_decks(seatmap):
            rendered = render_fn(deck, highlight=highlight, thick_border=thick_border)
            if rendered:
                rendered_decks.append(rendered)

        header_width = display_width(rendered_decks[0].splitlines()[0]) if rendered_decks else None
        header = self._format_header(seatmap, style=style, width=header_width) if show_header else ''
        output: list[str] = []
        if header:
            output.append(header)
        output.extend(rendered_decks)
        return '\n'.join(output)

    def _render_ascii_deck(self, deck: dict, *, highlight: str | None = None, thick_border: bool = False) -> str:
        rows, column_layout = self._build_seat_grid(deck)
        if not rows or not column_layout:
            return ''
        symbol_candidates = list(self.STATUS_SYMBOL.values()) + [
            self.WINDOW_AVAILABLE_SYMBOL,
            self.HIGHLIGHT_AVAILABLE_SYMBOL,
        ]
        symbol_width = max(display_width(symbol) for symbol in symbol_candidates)
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
        if not rows or not column_layout:
            return ''

        aisle_fill = '  '
        header_cells = [aisle_fill if col['is_aisle'] else (col['label'] or ' ') for col in column_layout]
        header = f"{'':>2} " + ''.join(header_cells)
        lines = [header]

        for row_name in sorted(rows, key=self._row_sort_key):
            seats_in_row = rows[row_name]
            row_cells = []
            for col in column_layout:
                if col['is_aisle']:
                    row_cells.append(aisle_fill)
                else:
                    row_cells.append(self._compact_seat_cell(seats_in_row.get(col['position'])))
            lines.append(f"{row_name:>2} " + ''.join(row_cells))

        return self._wrap_with_border(lines, highlight=highlight, thick_border=thick_border)

    def _compact_seat_cell(self, seat_info: dict | None) -> str:
        if not seat_info:
            return ' '
        availability = seat_info.get('availability') or 'UNKNOWN'
        is_window = seat_info.get('is_window', False)
        is_highlighted = seat_info.get('has_highlight_code', False)
        if availability == 'AVAILABLE' and is_highlighted:
            color_key = 'AVAILABLE_HIGHLIGHT'
        elif availability == 'AVAILABLE' and is_window:
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

        # First pass: collect columns and seat entries so we can infer windows even when codes are missing.
        seat_entries: list[tuple[int, dict]] = []
        for seat in seats:
            coords = seat.get('coordinates', {})
            column_position = coords.get('y')
            if column_position is None:
                continue
            seat_number = seat.get('number', '?')
            row_label, column_label = extract_row_and_column(seat_number)
            columns_by_position.setdefault(column_position, column_label)
            seat_entries.append((column_position, seat))

        if columns_by_position:
            ordered_cols = sorted(columns_by_position)
            window_positions = {ordered_cols[0], ordered_cols[-1]}
        else:
            window_positions = set()

        for column_position, seat in seat_entries:
            seat_number = seat.get('number', '?')
            row_label, _ = extract_row_and_column(seat_number)
            row_bucket = rows.setdefault(row_label, {})
            traveler_pricing = seat.get('travelerPricing', [])
            availability = traveler_pricing[0].get('seatAvailabilityStatus') if traveler_pricing else 'UNKNOWN'
            codes = seat.get('characteristicsCodes') or []
            has_window_code = 'W' in codes
            has_highlight_code = any(code in self.HIGHLIGHT_CHARACTERISTIC_CODES for code in codes)
            is_window = has_window_code or column_position in window_positions
            seat_symbol = self.STATUS_SYMBOL.get(availability, '?')
            if availability == 'AVAILABLE' and has_highlight_code:
                seat_symbol = self.HIGHLIGHT_AVAILABLE_SYMBOL
            elif availability == 'AVAILABLE' and is_window:
                seat_symbol = self.WINDOW_AVAILABLE_SYMBOL
            elif availability == 'OCCUPIED':
                seat_symbol = apply_color('fg_red', '×')
            row_bucket[column_position] = {
                'symbol': seat_symbol,
                'availability': availability,
                'is_window': is_window,
                'has_highlight_code': has_highlight_code,
            }

        column_layout = self._build_column_layout(columns_by_position)
        return rows, column_layout

    def _build_column_layout(self, columns_by_position: dict[int, str]) -> list[dict]:
        """Build ordered columns, inserting aisles when there are gaps or known splits."""
        ordered_columns = sorted(columns_by_position)
        layout: list[dict] = []
        last_label = None
        last_pos: int | None = None
        for pos in ordered_columns:
            label = columns_by_position[pos]
            gap = (pos - last_pos) if last_pos is not None else 0
            if last_pos is not None and (gap > 1 or self._has_aisle_between(last_label, label)):
                layout.append({'position': None, 'label': '', 'is_aisle': True})
            layout.append({'position': pos, 'label': label, 'is_aisle': False})
            last_label = label
            last_pos = pos
        return layout

    @staticmethod
    def _has_aisle_between(previous_label: str | None, next_label: str | None) -> bool:
        return (previous_label, next_label) in {
            ('B', 'D'),
            ('F', 'J'),
            ('G', 'J'),
        }

    def _format_header(self, seatmap: SeatMap, *, style: str, width: int | None = None) -> str:
        day_label = self._format_day_of_month(seatmap.departure_date)
        header = self._center_text(day_label, width)
        if style == 'compact':
            return apply_heatmap_header_color(header)
        return header

    @staticmethod
    def _format_day_of_month(date_str: str | None) -> str:
        if not date_str:
            return ''
        try:
            date_value = datetime.strptime(date_str, '%Y%m%d')
            return str(date_value.day)
        except ValueError:
            fallback = date_str[-2:]
            return fallback.lstrip('0') or fallback or date_str

    @staticmethod
    def _center_text(text: str, width: int | None) -> str:
        if not width or width <= 0:
            return text
        if display_width(text) >= width:
            return text
        return text.center(width)

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

    # Internal helpers for deck ordering
    def _sorted_decks(self, seatmap: SeatMap) -> list[dict]:
        """Sort decks so the lowest row appears first when multiple decks exist."""
        return sorted(seatmap.decks or [], key=self._deck_sort_key)

    def _deck_sort_key(self, deck: dict) -> tuple[int, int]:
        min_row = None
        for seat in deck.get('seats', []) or []:
            seat_number = seat.get('number')
            row_label, _ = extract_row_and_column(str(seat_number or ''))
            if row_label.isdigit():
                value = int(row_label)
            else:
                coords = seat.get('coordinates') or {}
                coord_row = coords.get('x')
                value = int(coord_row) if isinstance(coord_row, (int, float)) and coord_row > 0 else None
            if value is None:
                continue
            min_row = value if min_row is None else min(min_row, value)
        deck_number = deck.get('deckNumber')
        deck_hint = int(deck_number) if isinstance(deck_number, int) else 0
        return (min_row if min_row is not None else 10**9, deck_hint)


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
