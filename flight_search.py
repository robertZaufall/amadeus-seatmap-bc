import json
import os
import pickle
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from amadeus import Client, ResponseError
from dotenv import load_dotenv

load_dotenv()

#environment = "production"
#environment = "test"
#environment = "e2e"
environment = "e2e-pickle"


fixtures_dir = Path(__file__).parent / "test"


def extract_row_and_column(seat_number: str):
    row = ''.join(ch for ch in seat_number if ch.isdigit())
    column = ''.join(ch for ch in seat_number if ch.isalpha())
    return row, column


def char_display_width(character: str) -> int:
    """Return the display width of a single character."""
    return 2 if unicodedata.east_asian_width(character) in {'F', 'W'} else 1


def display_width(text: str) -> int:
    """Return the printable width of text accounting for wide characters."""
    return sum(char_display_width(ch) for ch in text or '')


def pad_to_width(text: str, width: int) -> str:
    """Pad or trim text so that its display width equals the provided width."""
    if width <= 0:
        return ''
    current_width = 0
    trimmed: list[str] = []
    for ch in text or '':
        ch_width = char_display_width(ch)
        if current_width + ch_width > width:
            break
        trimmed.append(ch)
        current_width += ch_width
    result = ''.join(trimmed)
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


class SeatMaps:
    STATUS_SYMBOL = {'AVAILABLE': '🟪', 'OCCUPIED': '❌', 'BLOCKED': '⬛'}
    WINDOW_AVAILABLE_SYMBOL = '🟩'

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
    def _build_seatmap(record: dict | None) -> SeatMap | None:
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
        return SeatMap(
            departure_date=departure_date,
            origin=origin,
            destination=destination,
            carrier=carrier,
            number=number,
            aircraft_code=aircraft_code,
            decks=decks,
            window_seats=window_seats,
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

    def render_map(self, seatmap: SeatMap) -> str:
        header = (f"{seatmap.departure_date} "
                  f"{seatmap.origin}{seatmap.destination} "
                  f"{seatmap.carrier}{seatmap.number}-{seatmap.aircraft_code} ")
        output = [f"\n{header}"]
        for deck in seatmap.decks:
            output.append(self._render_ascii_deck(deck))
        return '\n'.join(output)

    def _render_ascii_deck(self, deck: dict) -> str:
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
        bordered_lines = ['╭' + horizontal + '╮']
        for line in lines:
            padded = pad_line(line, content_width)
            bordered_lines.append(f"│ {padded} │")
        bordered_lines.append('╰' + horizontal + '╯')

        return '\n'.join(bordered_lines)

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
            return cls._build_seatmap(first_record)

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

        seatmap_request = {'data': [search_data[0]]}
        seatmap_response = amadeus.shopping.seatmaps.post(seatmap_request)
        seatmap_payload = seatmap_response.data or []
        first_record = seatmap_payload[0] if seatmap_payload else None
        return cls._build_seatmap(first_record)

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
else:
    travel_windows = [
        {
            "origin": "MUC",
            "destination": "BKK",
            "start_date": "2025-12-01",
            "end_date": "2025-12-20",
        },
        {
            "origin": "BKK",
            "destination": "MUC",
            "start_date": "2026-01-15",
            "end_date": "2026-01-25",
        },
    ]

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


def build_placeholder_block(date_key: str, width: int, height: int) -> list[str]:
    """Create a placeholder block for dates without seatmap data."""
    lines: list[str] = []
    lines.append(pad_to_width(f"{date_key} -- NO DATA --", width))
    if height == 1:
        return normalize_block(lines, width, height)

    inner_width = max(width - 2, 0)
    top_border = pad_to_width('╭' + ('─' * inner_width) + '╮', width)
    bottom_border = pad_to_width('╰' + ('─' * inner_width) + '╯', width)
    empty_body = pad_to_width('│' + (' ' * inner_width) + '│', width) if width >= 2 else pad_to_width('', width)
    lines.append(top_border)

    body_rows = max(height - 3, 0)
    if body_rows > 0 and inner_width > 0:
        message = 'NO DATA'
        trimmed_message = message[:inner_width]
        left_padding = (inner_width - len(trimmed_message)) // 2
        right_padding = inner_width - len(trimmed_message) - left_padding
        message_line = pad_to_width(
            '│' + (' ' * left_padding) + trimmed_message + (' ' * right_padding) + '│',
            width
        )
        lines.append(message_line)
        body_rows -= 1

    for _ in range(body_rows):
        lines.append(empty_body)

    lines.append(bottom_border)
    return normalize_block(lines, width, height)


def print_weekly_layout(seatmaps_obj: SeatMaps, seatmaps_by_date: dict[str, SeatMap]) -> None:
    """Print seatmaps grouped by week, filling missing days with placeholders."""
    if not seatmaps_by_date:
        return

    rendered_blocks: dict[str, list[str]] = {}
    max_width = 0
    max_height = 0
    for date_key, seatmap_obj in seatmaps_by_date.items():
        block_lines = seatmaps_obj.render_map(seatmap_obj).splitlines()
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
            print()

        for line_idx in range(max_height):
            print('  '.join(block[line_idx] for block in weekly_blocks))
        print()

        previous_week_signature = week_signature
        current += timedelta(days=7)

print("\n\n")
print_weekly_layout(seatmaps, seatmaps_by_date)

if seatmaps_by_date:
    print("\nAvailable window seats by date:")
    previous_destination: str | None = None
    for date_key in sorted(seatmaps_by_date):
        seatmap = seatmaps_by_date[date_key]
        if previous_destination and seatmap.destination != previous_destination:
            print()
            print()
        seats = seatmap.window_seats
        if seats:
            sorted_seats = ', '.join(sorted(seats, key=window_seat_sort_key))
            print(f"{date_key}: {sorted_seats}")
        previous_destination = seatmap.destination

print("\n\n")
