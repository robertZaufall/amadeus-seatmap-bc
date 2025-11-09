import json
import os
import pickle
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path

from amadeus import Client, ResponseError
from dotenv import load_dotenv

from config import (
    ANSI_RESET as CONFIG_ANSI_RESET,
    BORDER_COLORS,
    COMPACT_BACKGROUND_COLORS,
    COMPACT_SYMBOL_COLORS,
    COMPACT_SYMBOLS,
    ENVIRONMENT,
    HEATMAP_HEADER_COLOR,
    HEATMAP_COLORS,
    HEATMAP_SYMBOLS,
    CURRENCY_SYMBOLS,
    HEATMAP_EMPHASIS_STYLES,
    SEATMAP_OUTPUT_STYLE,
    SHOW_SEATMAP_PRICE,
    STATIC_LABELS,
    STATUS_SYMBOLS,
    TRAVEL_WINDOWS,
    WINDOW_AVAILABLE_SYMBOL as CONFIG_WINDOW_AVAILABLE_SYMBOL,
)
from colors import apply as apply_color, resolve as resolve_color, ANSI_RESET as COLORS_ANSI_RESET

load_dotenv()

environment = ENVIRONMENT
travel_windows = TRAVEL_WINDOWS

travel_window_ranges = [
    (
        datetime.fromisoformat(window["start_date"]).date(),
        datetime.fromisoformat(window["end_date"]).date(),
    )
    for window in travel_windows
]

route_travel_windows: dict[str, list[tuple[datetime.date, datetime.date]]] = defaultdict(list)
for window in travel_windows:
    start = datetime.fromisoformat(window["start_date"]).date()
    end = datetime.fromisoformat(window["end_date"]).date()
    route = f"{window['origin']}->{window['destination']}"
    route_travel_windows[route].append((start, end))

fixtures_dir = Path(__file__).parent / "test"
ANSI_ESCAPE_RE = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')


def _load_flight_offer_fixtures() -> dict[str, dict]:
    """Return flight offer fixtures keyed by their id."""
    flight_offer_path = fixtures_dir / "flight-offer.json"
    if not flight_offer_path.exists():
        return {}

    with open(flight_offer_path, encoding="utf-8") as fixture:
        payload = json.load(fixture)

    if isinstance(payload, dict) and "data" in payload:
        offers = payload["data"]
    elif isinstance(payload, list):
        offers = payload
    elif isinstance(payload, dict):
        offers = [payload]
    else:
        return {}

    offer_map: dict[str, dict] = {}
    for offer in offers:
        if not isinstance(offer, dict):
            continue
        offer_id = str(offer.get("id") or "")
        if offer_id:
            offer_map[offer_id] = offer
    return offer_map


def extract_row_and_column(seat_number: str):
    row = ''.join(ch for ch in seat_number if ch.isdigit())
    column = ''.join(ch for ch in seat_number if ch.isalpha())
    return row, column


def char_display_width(character: str) -> int:
    """Return the display width of a single character."""
    return 2 if unicodedata.east_asian_width(character) in {'F', 'W'} else 1

def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    if not text:
        return ''
    return ANSI_ESCAPE_RE.sub('', text)


def display_width(text: str) -> int:
    """Return the printable width of text accounting for wide characters."""
    visible = strip_ansi(text or '')
    return sum(char_display_width(ch) for ch in visible)


def resolve_seatmap_style(value: str | None) -> str:
    """Normalize the configured seatmap style to either 'ascii' or 'compact'."""
    normalized = (value or '').strip().lower()
    return 'compact' if normalized == 'compact' else 'ascii'


EMPHASIS_CODE_MAP = {
    'bold': '\033[1m',
    'italic': '\033[3m',
}


def apply_emphasis_styles(text: str, *, enabled: bool) -> str:
    """Wrap text with configured ANSI emphasis codes when enabled."""
    if not enabled or not text:
        return text
    segments: list[str] = []
    if HEATMAP_EMPHASIS_STYLES.get('italic'):
        segments.append(EMPHASIS_CODE_MAP['italic'])
    if HEATMAP_EMPHASIS_STYLES.get('bold'):
        segments.append(EMPHASIS_CODE_MAP['bold'])
    if not segments:
        return text
    prefix = ''.join(segments)
    return f"{prefix}{text}{CONFIG_ANSI_RESET}"


def apply_italic_only(text: str) -> str:
    """Wrap text with ANSI italic without inheriting bold config."""
    if not text:
        return text
    italic_code = EMPHASIS_CODE_MAP.get('italic')
    if not italic_code:
        return text
    return f"{italic_code}{text}{CONFIG_ANSI_RESET}"


def apply_bold_italic(text: str) -> str:
    """Apply both bold and italic emphasis regardless of config."""
    if not text:
        return text
    bold_code = EMPHASIS_CODE_MAP.get('bold')
    italic_code = EMPHASIS_CODE_MAP.get('italic')
    segments = [segment for segment in (italic_code, bold_code) if segment]
    if not segments:
        return text
    prefix = ''.join(segments)
    return f"{prefix}{text}{CONFIG_ANSI_RESET}"


def pad_to_width(text: str, width: int) -> str:
    """Pad or trim text so that its display width equals the provided width."""
    if width <= 0:
        return ''
    current_width = 0
    trimmed_parts: list[str] = []
    text = text or ''
    idx = 0
    while idx < len(text):
        if text[idx] == '\x1b':
            match = ANSI_ESCAPE_RE.match(text, idx)
            if match:
                trimmed_parts.append(match.group(0))
                idx = match.end()
                continue
        ch = text[idx]
        ch_width = char_display_width(ch)
        if current_width + ch_width > width:
            break
        trimmed_parts.append(ch)
        current_width += ch_width
        idx += 1
    result = ''.join(trimmed_parts)
    if current_width < width:
        result += ' ' * (width - current_width)
    return result

def pad_to_width_centered(text: str, width: int) -> str:
    """Pad text to the provided width while keeping the content centered."""
    if width <= 0:
        return ''
    trimmed = pad_to_width(text, width)
    trimmed_content = trimmed.rstrip()
    content_width = display_width(trimmed_content)
    if content_width >= width:
        return trimmed_content
    padding = width - content_width
    left_padding = padding // 2
    right_padding = padding - left_padding
    return f"{' ' * left_padding}{trimmed_content}{' ' * right_padding}"

WEEKDAY_SHORT_NAMES = ('Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su')

def weekday_short_name(value: date) -> str:
    """Return a two-letter weekday abbreviation starting on Monday."""
    return WEEKDAY_SHORT_NAMES[value.weekday()]

@dataclass
class SeatMap:
    departure_date: str
    origin: str
    destination: str
    carrier: str
    number: str
    aircraft_code: str
    decks: list
    window_seats: list[str] = field(default_factory=list)
    price_total: str | None = None
    price_currency: str | None = None

    def formatted_total_price(self, *, rounded: bool = False) -> str | None:
        display_total = self.price_total
        if rounded and display_total:
            try:
                display_total = str(Decimal(display_total).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
            except (InvalidOperation, ValueError):
                pass
        currency_code = (self.price_currency or '').upper()
        currency_symbol = CURRENCY_SYMBOLS.get(currency_code)
        if display_total and currency_code:
            if currency_symbol:
                return f"{currency_symbol}{display_total}"
            return f"{display_total} {currency_code}"
        if display_total:
            return display_total
        return None


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

    @staticmethod
    def _extract_price_fields(price_info: dict | None) -> tuple[str | None, str | None]:
        if not isinstance(price_info, dict):
            return None, None
        total = price_info.get('grandTotal') or price_info.get('total')
        currency = price_info.get('currency')
        return total, currency

    @staticmethod
    def _build_seatmap(record: dict | None, price_info: dict | None = None) -> SeatMap | None:
        if record is None:
            return None
        departure_info = record.get('departure', {})
        arrival_info = record.get('arrival', {})
        departure_date = departure_info.get('at', '').split('T')[0].replace('-', '')
        origin = departure_info.get('iataCode', '')
        destination = arrival_info.get('iataCode', '')
        carrier = record.get('carrierCode', '')
        number = record.get('number', '')
        aircraft_code = record.get('aircraft', {}).get('code', 'N/A')
        decks = record.get('decks', [])
        window_seats = SeatMaps._collect_available_window_seats(decks)
        total_price, price_currency = SeatMaps._extract_price_fields(price_info)
        return SeatMap(
            departure_date=departure_date,
            origin=origin,
            destination=destination,
            carrier=carrier,
            number=number,
            aircraft_code=aircraft_code,
            decks=decks,
            window_seats=window_seats,
            price_total=total_price,
            price_currency=price_currency,
        )

    @staticmethod
    def _collect_available_window_seats(decks: list[dict]) -> list[str]:
        available = []
        for deck in decks:
            for seat in deck.get('seats', []):
                traveler_pricing = seat.get('travelerPricing', [])
                availability = traveler_pricing[0].get('seatAvailabilityStatus') if traveler_pricing else None
                if availability != 'AVAILABLE':
                    continue
                if 'W' not in seat.get('characteristicsCodes', []):
                    continue
                available.append(seat.get('number'))
        return available

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
            secondary_line = self._build_compact_header_meta_line(
                weekday_label=self._weekday_label(seatmap.departure_date),
                price_label=seatmap.formatted_total_price(rounded=True) or "N/A",
                aircraft_label=seatmap.aircraft_code or '',
                layout=layout,
            )
            lines = [line for line in (primary_line, secondary_line) if line]
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

        # Use helper to apply color+reset consistently
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
        # left vertical border, content, right vertical border
        box_lines.append(f"{apply_color(color, '│')} {line} {apply_color(color, '│')}")
    box_lines.append(apply_color(color, f"╰{horizontal}╯"))
    return box_lines

    @classmethod
    def fetch(
        cls,
        *,
        environment: str,
        origin_location_code: str,
        destination_location_code: str,
        departure_date: str,
        travel_class: str,
        non_stop: str,
        included_airline_codes: str
    ) -> SeatMap | None:
        """Retrieve seat map data for the provided criteria."""
        if environment == "e2e":
            with open(fixtures_dir / "seatmap.json", encoding="utf-8") as fixture:
                seatmap_fixtures = json.load(fixture)
            first_record = seatmap_fixtures[0] if seatmap_fixtures else None
            price_lookup = _load_flight_offer_fixtures()
            flight_offer_id = first_record.get('flightOfferId') if isinstance(first_record, dict) else None
            offer = price_lookup.get(str(flight_offer_id)) if flight_offer_id is not None else None
            price_info = offer.get('price') if isinstance(offer, dict) else None
            return cls._build_seatmap(first_record, price_info)

        if environment not in {"test", "production"}:
            raise ValueError(f"Unsupported environment '{environment}'")

        if environment == "test":
            client_id = os.getenv("TEST_AMADEUS_CLIENT_ID")
            client_secret = os.getenv("TEST_AMADEUS_CLIENT_SECRET")
        else:
            client_id = os.getenv("AMADEUS_CLIENT_ID")
            client_secret = os.getenv("AMADEUS_CLIENT_SECRET")

        amadeus = Client(
            client_id=client_id,
            client_secret=client_secret,
            hostname=environment
        )

        search_response = amadeus.shopping.flight_offers_search.get(
            originLocationCode=origin_location_code,
            destinationLocationCode=destination_location_code,
            departureDate=departure_date,
            travelClass=travel_class,
            nonStop=non_stop,
            includedAirlineCodes=included_airline_codes,
            adults=1
        )
        search_data = search_response.data or []
        if not search_data:
            return None

        first_offer = search_data[0]
        price_info = first_offer.get('price') if isinstance(first_offer, dict) else None
        seatmap_request = {'data': [first_offer]}
        seatmap_response = amadeus.shopping.seatmaps.post(seatmap_request)
        seatmap_payload = seatmap_response.data or []
        first_record = seatmap_payload[0] if seatmap_payload else None
        return cls._build_seatmap(first_record, price_info)

def iter_dates(start_date: str, end_date: str):
    current = datetime.fromisoformat(start_date)
    stop = datetime.fromisoformat(end_date)
    while current <= stop:
        yield current.strftime('%Y-%m-%d')
        current += timedelta(days=1)


seatmaps = SeatMaps()

if environment == "e2e-pickle":
    pickle_path = fixtures_dir / "seatmaps.pkl"
    with open(pickle_path, "rb") as pickled_seatmaps:
        seatmaps.seatmaps = pickle.load(pickled_seatmaps)
    for seatmap_obj in seatmaps:
        if not hasattr(seatmap_obj, "price_total"):
            seatmap_obj.price_total = None
        if not hasattr(seatmap_obj, "price_currency"):
            seatmap_obj.price_currency = None
else:
    for window in travel_windows:
        for departure_date in iter_dates(window["start_date"], window["end_date"]):
            try:
                seatmap = SeatMaps.fetch(
                    environment=environment,
                    origin_location_code=window["origin"],
                    destination_location_code=window["destination"],
                    departure_date=departure_date,
                    travel_class='BUSINESS',
                    non_stop='true',
                    included_airline_codes='TG'
                )
            except ResponseError as error:
                raise error

            seatmaps.add(seatmap)

    if not seatmaps:
        print("No seat maps available")
        exit()

    pickle_path = fixtures_dir / "seatmaps.pkl"
    with open(pickle_path, "wb") as pickled_seatmaps:
        pickle.dump(seatmaps.seatmaps, pickled_seatmaps)

seatmaps_by_date = {
    seatmap_obj.departure_date: seatmap_obj
    for seatmap_obj in seatmaps
}

def parse_total_price(value: str | None) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(value)
    except (InvalidOperation, ValueError):
        return None


def compute_best_price_by_route(seatmaps_obj: SeatMaps) -> dict[tuple[str, str], Decimal]:
    best: dict[tuple[str, str], Decimal] = {}
    for seatmap_obj in seatmaps_obj:
        price = parse_total_price(seatmap_obj.price_total)
        if price is None:
            continue
        route = (seatmap_obj.origin, seatmap_obj.destination)
        existing = best.get(route)
        if existing is None or price < existing:
            best[route] = price
    return best


def compute_worst_price_by_route(seatmaps_obj: SeatMaps) -> dict[tuple[str, str], Decimal]:
    worst: dict[tuple[str, str], Decimal] = {}
    for seatmap_obj in seatmaps_obj:
        price = parse_total_price(seatmap_obj.price_total)
        if price is None:
            continue
        route = (seatmap_obj.origin, seatmap_obj.destination)
        existing = worst.get(route)
        if existing is None or price > existing:
            worst[route] = price
    return worst


def has_best_price_for_route(seatmap_obj: SeatMap, price_lookup: dict[tuple[str, str], Decimal]) -> bool:
    price = parse_total_price(seatmap_obj.price_total)
    if price is None:
        return False
    route = (seatmap_obj.origin, seatmap_obj.destination)
    best_price = price_lookup.get(route)
    return best_price is not None and price == best_price


def has_worst_price_for_route(seatmap_obj: SeatMap, price_lookup: dict[tuple[str, str], Decimal]) -> bool:
    price = parse_total_price(seatmap_obj.price_total)
    if price is None:
        return False
    route = (seatmap_obj.origin, seatmap_obj.destination)
    worst_price = price_lookup.get(route)
    return worst_price is not None and price == worst_price


best_price_by_route = compute_best_price_by_route(seatmaps)
worst_price_by_route = compute_worst_price_by_route(seatmaps)


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


def build_heatmap_entries(seatmaps_by_date: dict[str, SeatMap]) -> dict[str, dict[str, Decimal]]:
    route_entries: dict[str, dict[str, Decimal]] = defaultdict(dict)
    for window in travel_windows:
        route_key = f"{window['origin']}->{window['destination']}"
        for date_iso in iter_dates(window["start_date"], window["end_date"]):
            date_key = date_iso.replace('-', '')
            seatmap = seatmaps_by_date.get(date_key)
            if seatmap is None or not seatmap.window_seats:
                continue
            price = parse_total_price(seatmap.price_total)
            if price is None:
                continue
            route_entries[route_key][date_key] = price
    return route_entries


def build_price_entries_all_dates(seatmaps_by_date: dict[str, SeatMap]) -> dict[str, dict[str, Decimal]]:
    """Build price entries for every seatmap date regardless of window-seat availability."""
    route_entries: dict[str, dict[str, Decimal]] = defaultdict(dict)
    for window in travel_windows:
        route_key = f"{window['origin']}->{window['destination']}"
        for date_iso in iter_dates(window["start_date"], window["end_date"]):
            date_key = date_iso.replace('-', '')
            seatmap = seatmaps_by_date.get(date_key)
            if seatmap is None:
                continue
            price = parse_total_price(seatmap.price_total)
            if price is None:
                continue
            route_entries[route_key][date_key] = price
    return route_entries


def build_heatmap_price_stats(entries_by_route: dict[str, dict[str, Decimal]]) -> dict[str, tuple[Decimal, Decimal]]:
    stats: dict[str, tuple[Decimal, Decimal]] = {}
    for route, entries in entries_by_route.items():
        if not entries:
            continue
        prices = list(entries.values())
        stats[route] = (min(prices), max(prices))
    return stats


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


def apply_heatmap_header_color(text: str) -> str:
    """Color header cells using the configured darker grey tone."""
    if not text or not HEATMAP_HEADER_COLOR:
        return text
    return apply_color(HEATMAP_HEADER_COLOR, text)


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


def format_roundtrip_price_heatmap(
    heatmap_entries: dict[str, dict[str, Decimal]],
    *,
    title: str | None = None,
    highlight_entries: dict[str, dict[str, Decimal]] | None = None,
    emphasize_highlights: bool = True,
    highlight_emphasis: str = 'config',
) -> list[str]:
    """Return a combined outbound/return heatmap covering the first two travel windows."""
    if len(travel_windows) < 2 or not heatmap_entries:
        return []

    outbound_window, return_window = travel_windows[:2]
    outbound_route = f"{outbound_window['origin']}->{outbound_window['destination']}"
    return_route = f"{return_window['origin']}->{return_window['destination']}"
    outbound_entries = heatmap_entries.get(outbound_route, {})
    return_entries = heatmap_entries.get(return_route, {})

    def build_axis(
        window: dict,
        route_entries: dict[str, Decimal],
        highlight_lookup: dict[str, Decimal] | None,
    ) -> list[tuple[str, date, Decimal | None, bool]]:
        axis: list[tuple[str, date, Decimal | None, bool]] = []
        for date_iso in iter_dates(window["start_date"], window["end_date"]):
            date_obj = datetime.fromisoformat(date_iso).date()
            date_key = date_obj.strftime('%Y%m%d')
            has_window = bool(highlight_lookup and date_key in highlight_lookup)
            axis.append((date_key, date_obj, route_entries.get(date_key), has_window))
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
        return []

    combined_matrix: list[list[Decimal | None]] = []
    combined_values: list[Decimal] = []
    for _, _, return_price, _ in return_axis:
        row: list[Decimal | None] = []
        for _, _, outbound_price, _ in outbound_axis:
            if outbound_price is not None and return_price is not None:
                combined = outbound_price + return_price
                combined_values.append(combined)
                row.append(combined)
            else:
                row.append(None)
        combined_matrix.append(row)

    min_combined = min(combined_values) if combined_values else None
    max_combined = max(combined_values) if combined_values else None

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
        _, _, _, return_has_window = return_axis[row_idx]
        rendered_row: list[str] = []
        for col_idx, value in enumerate(row):
            _, _, _, outbound_has_window = outbound_axis[col_idx]
            emphasize_cell = should_emphasize and outbound_has_window and return_has_window
            rendered_row.append(format_cell(value, emphasize=emphasize_cell))
        rendered_rows.append(rendered_row)
    def format_date_label(value: date) -> str:
        return value.strftime('%m%d')

    row_labels = [f"{weekday_short_name(date_value)} {format_date_label(date_value)}" for _, date_value, _, _ in return_axis]
    column_labels = [format_date_label(date_value) for _, date_value, _, _ in outbound_axis]
    column_weekdays = [weekday_short_name(date_value) for _, date_value, _, _ in outbound_axis]
    column_years: list[str] = []
    previous_year: str | None = None
    for _, date_value, _, _ in outbound_axis:
        year_str = date_value.strftime('%Y')
        if year_str != previous_year:
            column_years.append(year_str)
            previous_year = year_str
        else:
            column_years.append('')

    row_years: list[str] = []
    previous_row_year: str | None = None
    for _, date_value, _, _ in return_axis:
        year_str = date_value.strftime('%Y')
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
        outbound_route=outbound_route,
        return_route=return_route,
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
    heatmap_entries = build_heatmap_entries(seatmaps_by_date)
    heatmap_stats = build_heatmap_price_stats(heatmap_entries)
    all_price_entries = build_price_entries_all_dates(seatmaps_by_date)
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

    window_roundtrip_heatmap = format_roundtrip_price_heatmap(
        heatmap_entries,
        title=STATIC_LABELS['roundtrip_window_title'],
    )
    if window_roundtrip_heatmap:
        print()
        for line in window_roundtrip_heatmap:
            print(line)

    all_price_roundtrip_heatmap = format_roundtrip_price_heatmap(
        all_price_entries,
        title=STATIC_LABELS['roundtrip_all_title'],
        highlight_entries=heatmap_entries,
        emphasize_highlights=True,
        highlight_emphasis='bold_italic',
    )
    if all_price_roundtrip_heatmap:
        for line in all_price_roundtrip_heatmap:
            print(line)

print("\n")
