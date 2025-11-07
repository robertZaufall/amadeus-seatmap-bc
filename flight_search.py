import json
import os
import unicodedata
from pathlib import Path

from amadeus import Client, ResponseError
from dotenv import load_dotenv

load_dotenv()

#environment = "production"
#environment = "test"
environment = "e2e"

amadeus = None
if environment == "test":
    amadeus = Client(
        client_id=os.getenv("TEST_AMADEUS_CLIENT_ID"),
        client_secret=os.getenv("TEST_AMADEUS_CLIENT_SECRET"),
        hostname=environment
    )
elif environment == "production":
    amadeus = Client(
        client_id=os.getenv("AMADEUS_CLIENT_ID"),
        client_secret=os.getenv("AMADEUS_CLIENT_SECRET"),
        hostname='production'
    )
elif environment != "e2e":
    raise ValueError(f"Unsupported environment '{environment}'")

fixtures_dir = Path(__file__).parent / "test"

try:
    if environment == "e2e":
        with open(fixtures_dir / "flight-offer.json", encoding="utf-8") as fixture:
            flight = json.load(fixture)
        with open(fixtures_dir / "seatmap.json", encoding="utf-8") as fixture:
            seatmap_data = json.load(fixture)
    else:
        search_response = amadeus.shopping.flight_offers_search.get(
            originLocationCode='MUC',
            destinationLocationCode='BKK',
            departureDate='2025-12-15',
            travelClass='BUSINESS',
            nonStop='true',
            includedAirlineCodes='TG',
            adults=1
        )
        search_data = search_response.data
        if not search_data:
            print("No flight offers found")
            exit()

        flight = search_data[0]

        seatmap_request = {'data': [flight]}
        seatmap_response = amadeus.shopping.seatmaps.post(seatmap_request)
        seatmap_data = seatmap_response.data

    #print(json.dumps(flight, indent=2))
    #print(json.dumps(seatmap_data, indent=2))

    def extract_row_and_column(seat_number: str):
        row = ''.join(ch for ch in seat_number if ch.isdigit())
        column = ''.join(ch for ch in seat_number if ch.isalpha())
        return row, column

    def render_ascii_deck(deck: dict) -> str:
        seats = deck.get('seats', [])
        columns_by_position = {}
        rows = {}
        status_symbol = {'AVAILABLE': '🟩', 'OCCUPIED': '🟥', 'BLOCKED': '⬛'}

        def display_width(text: str) -> int:
            width = 0
            for ch in text or '':
                width += 2 if unicodedata.east_asian_width(ch) in {'F', 'W'} else 1
            return width

        symbol_width = max((display_width(symbol) for symbol in status_symbol.values()), default=1)
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
            row_bucket[column_position] = status_symbol.get(availability, '?')

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
        header = f"{'Row':>3} " + ''.join(header_cells).replace('A B   D E F G   J K', 'A B   D E  F G  J K')
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

        #lines.append("Legend: 🟩=Available, 🟥=Occupied, ⬛=Blocked, blank=No seat")
        return '\n'.join(lines)

    def collect_available_window_seats(deck: dict):
        available = []
        for seat in deck.get('seats', []):
            traveler_pricing = seat.get('travelerPricing', [])
            availability = traveler_pricing[0].get('seatAvailabilityStatus') if traveler_pricing else None
            if availability != 'AVAILABLE':
                continue
            if 'W' not in seat.get('characteristicsCodes', []):
                continue
            available.append(seat.get('number'))
        return available

    window_seats = []
    for seatmap in seatmap_data:
        aircraft_code = seatmap.get('aircraft', {}).get('code', 'N/A')
        display_aircraft = 'A395' if aircraft_code == '359' else aircraft_code
        departure_info = seatmap.get('departure', {})
        print("\n"
              f"{departure_info.get('at', '').split('T')[0].replace('-', '')} "
              f"{departure_info.get('iataCode')}{seatmap.get('arrival', {}).get('iataCode')} "
              f"{seatmap.get('carrierCode')}{seatmap.get('number')}-"
              f"{aircraft_code} "
              #f"{display_aircraft} "
        )
        for deck in seatmap.get('decks', []):
            print(render_ascii_deck(deck))
            window_seats.extend(collect_available_window_seats(deck))

    if window_seats:
        print("\nAvailable window seats: " + ', '.join(sorted(window_seats, key=lambda x: (int(''.join(filter(str.isdigit, x)) or 0), x))))

except ResponseError as error:
    raise error
