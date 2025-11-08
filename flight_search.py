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

load_dotenv()

#environment = "production"
#environment = "test"
#environment = "e2e"
environment = "e2e-pickle"

travel_windows = [
    {
        "origin": "MUC",
        "destination": "BKK",
        "start_date": "2025-11-24",
        "end_date": "2025-12-20",
    },
    {
        "origin": "BKK",
        "destination": "MUC",
        "start_date": "2026-01-15",
        "end_date": "2026-01-30",
    },
]

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
        if display_total and self.price_currency:
            return f"{display_total} {self.price_currency}"
        if display_total:
            return display_total
        return None


class SeatMaps:
    STATUS_SYMBOL = {'AVAILABLE': '🟪', 'OCCUPIED': '❌', 'BLOCKED': '⬛'}
    WINDOW_AVAILABLE_SYMBOL = '🟩'
    BORDER_COLOR_DEFAULT = '\033[90m'
    BORDER_COLOR_HIGHLIGHT = '\033[32m'
    ANSI_RESET = '\033[0m'

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

    def render_map(self, seatmap: SeatMap, *, highlight_border: bool = False) -> str:
        header = (f"{seatmap.departure_date} "
                  f"{seatmap.origin}{seatmap.destination} "
                  f"{seatmap.carrier}{seatmap.number}-{seatmap.aircraft_code} ")
        output = [f"\n{header}"]
        for deck in seatmap.decks:
            output.append(self._render_ascii_deck(deck, highlight_border=highlight_border))
        return '\n'.join(output)

    def _render_ascii_deck(self, deck: dict, *, highlight_border: bool = False) -> str:
        seats = deck.get('seats', [])
        columns_by_position = {}
        rows = {}

        symbol_width = max(
            [display_width(symbol) for symbol in self.STATUS_SYMBOL.values()] + [display_width(self.WINDOW_AVAILABLE_SYMBOL)]
        )
        seat_column_width = max(symbol_width, 1)
        aisle_column_width = max(1, seat_column_width // 2) + 1

        def format_cell(value: str, width: int) -> str:
            value = value or ''
            pad = max(width - display_width(value), 0)
            return value + (' ' * pad)

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
            seat_symbol = self.STATUS_SYMBOL.get(availability, '?')
            if availability == 'AVAILABLE' and 'W' in seat.get('characteristicsCodes', []):
                seat_symbol = self.WINDOW_AVAILABLE_SYMBOL
            row_bucket[column_position] = seat_symbol

        ordered_columns = sorted(columns_by_position)
        display_columns = []
        last_label = None
        for pos in ordered_columns:
            label = columns_by_position[pos]
            if last_label == 'B' and label == 'D':
                display_columns.append((None, '', aisle_column_width))
            if last_label == 'G' and label == 'J':
                display_columns.append((None, '', aisle_column_width))
            display_columns.append((pos, label, seat_column_width))
            last_label = label

        header_cells = [format_cell(col_label, width) for _, col_label, width in display_columns]
        header = f"{'Row':>3} " + ''.join(header_cells).replace('A B   D E F G   J K', ' A B  D E  F G  J K')
        lines = [header]

        def sort_key(row_name: str):
            return (0, int(row_name)) if row_name.isdigit() else (1, row_name)

        for row_name in sorted(rows, key=sort_key):
            seats_in_row = rows[row_name]
            cells = []
            for pos, _, width in display_columns:
                if pos is None:
                    cells.append(format_cell('', width))
                else:
                    cells.append(format_cell(seats_in_row.get(pos, ' '), width))
            lines.append(f"{row_name:>3} " + ''.join(cells))

        def pad_line(text: str, width: int) -> str:
            pad = max(width - display_width(text), 0)
            return text + (' ' * pad)

        content_width = max((display_width(line) for line in lines), default=0)
        horizontal = '─' * (content_width + 2)
        border_color = self.BORDER_COLOR_HIGHLIGHT if highlight_border else self.BORDER_COLOR_DEFAULT
        border_reset = self.ANSI_RESET
        bordered_lines = [f"{border_color}╭{horizontal}╮{border_reset}"]
        left_border = f"{border_color}│{border_reset}"
        right_border = f"{border_color}│{border_reset}"
        for line in lines:
            padded = pad_line(line, content_width)
            bordered_lines.append(f"{left_border} {padded} {right_border}")
        bordered_lines.append(f"{border_color}╰{horizontal}╯{border_reset}")

        return '\n'.join(bordered_lines)


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
    reset = SeatMaps.ANSI_RESET
    box_lines = [f"{color}╭{horizontal}╮{reset}"]
    for line in padded_lines:
        box_lines.append(f"{color}│{reset} {line} {color}│{reset}")
    box_lines.append(f"{color}╰{horizontal}╯{reset}")
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


def has_best_price_for_route(seatmap_obj: SeatMap, price_lookup: dict[tuple[str, str], Decimal]) -> bool:
    price = parse_total_price(seatmap_obj.price_total)
    if price is None:
        return False
    route = (seatmap_obj.origin, seatmap_obj.destination)
    best_price = price_lookup.get(route)
    return best_price is not None and price == best_price


best_price_by_route = compute_best_price_by_route(seatmaps)


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


def build_heatmap_price_stats(entries_by_route: dict[str, dict[str, Decimal]]) -> dict[str, tuple[Decimal, Decimal]]:
    stats: dict[str, tuple[Decimal, Decimal]] = {}
    for route, entries in entries_by_route.items():
        if not entries:
            continue
        prices = list(entries.values())
        stats[route] = (min(prices), max(prices))
    return stats


HEATMAP_SYMBOL_MIN = '🟩'
HEATMAP_SYMBOL_DEFAULT = '🟨'
HEATMAP_SYMBOL_MAX = '🟥'
HEATMAP_COLOR_MIN = SeatMaps.BORDER_COLOR_HIGHLIGHT
HEATMAP_COLOR_DEFAULT = '\033[33m'
HEATMAP_COLOR_MAX = '\033[31m'
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
    return f"{color}{symbol}{SeatMaps.ANSI_RESET}"


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
        weekday_names = ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su']
        day_header = ' '.join(pad_to_width(name, HEATMAP_CELL_WIDTH) for name in weekday_names)
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
        entries = route_lines[route] or ['No window seats']
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
    best_price_by_route: dict[tuple[str, str], Decimal] | None = None
) -> None:
    """Print seatmaps grouped by week, filling missing days with placeholders."""
    if not seatmaps_by_date:
        return

    rendered_blocks: dict[str, list[str]] = {}
    max_width = 0
    max_height = 0
    for date_key, seatmap_obj in seatmaps_by_date.items():
        highlight_border = (
            best_price_by_route is not None and has_best_price_for_route(seatmap_obj, best_price_by_route)
        )
        block_lines = seatmaps_obj.render_map(seatmap_obj, highlight_border=highlight_border).splitlines()
        while block_lines and not block_lines[0].strip():
            block_lines = block_lines[1:]
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

print("\n\n")
print_weekly_layout(seatmaps, seatmaps_by_date, best_price_by_route=best_price_by_route)

if seatmaps_by_date:
    print("\nAvailable window seats by date:")
    heatmap_entries = build_heatmap_entries(seatmaps_by_date)
    heatmap_stats = build_heatmap_price_stats(heatmap_entries)
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
        availability_by_route.setdefault(route_key, []).append(
            f"{symbol_prefix} {formatted_date} ({price_text}): {sorted_seats}"
        )

    if availability_by_route:
        ordered_routes = sorted(route_first_date, key=route_first_date.get)
        render_availability_boxes(
            availability_by_route,
            route_order=ordered_routes,
            heatmap_entries=heatmap_entries,
            heatmap_stats=heatmap_stats,
        )

print("\n")
