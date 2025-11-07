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
    STATUS_SYMBOL = {'AVAILABLE': '🟩', 'OCCUPIED': '🟥', 'BLOCKED': '⬛'}

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

        def display_width(text: str) -> int:
            width = 0
            for ch in text or '':
                width += 2 if unicodedata.east_asian_width(ch) in {'F', 'W'} else 1
            return width

        symbol_width = max((display_width(symbol) for symbol in self.STATUS_SYMBOL.values()), default=1)
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
            row_bucket[column_position] = self.STATUS_SYMBOL.get(availability, '?')

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

seatmaps_by_date = {}
window_seats = []
for seatmap_obj in seatmaps:
    seatmaps_by_date[seatmap_obj.departure_date] = seatmap_obj

for date_key in sorted(seatmaps_by_date):
    seatmap_obj = seatmaps_by_date[date_key]
    print(seatmaps.render_map(seatmap_obj))
    window_seats.extend(seatmap_obj.window_seats)

if window_seats:
    print("\nAvailable window seats: " + ', '.join(sorted(window_seats, key=lambda x: (int(''.join(filter(str.isdigit, x)) or 0), x))))
