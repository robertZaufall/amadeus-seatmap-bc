from __future__ import annotations

import json
import os
import pickle
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable, Iterator

from amadeus import Client, ResponseError

from config import CURRENCY_SYMBOLS


FIXTURES_DIR = Path(__file__).parent / "test"
SEATMAPS_PICKLE_PATH = FIXTURES_DIR / "seatmaps.pkl"
ROUNDTRIP_DB_PATH = Path(__file__).parent / "roundtrip_prices.db"


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
    price_timestamp: datetime | None = None

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


def _load_flight_offer_fixtures(fixtures_dir: Path = FIXTURES_DIR) -> dict[str, dict]:
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


def _extract_price_fields(price_info: dict | None) -> tuple[str | None, str | None]:
    if not isinstance(price_info, dict):
        return None, None
    total = price_info.get('grandTotal') or price_info.get('total')
    currency = price_info.get('currency')
    return total, currency


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


def _build_seatmap(
    record: dict | None,
    price_info: dict | None = None,
    *,
    captured_at: datetime | None = None,
) -> SeatMap | None:
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
    window_seats = _collect_available_window_seats(decks)
    timestamp = captured_at or datetime.utcnow()
    total_price, price_currency = _extract_price_fields(price_info)
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
        price_timestamp=timestamp,
    )


def iter_dates(start_date: str, end_date: str) -> Iterator[str]:
    current = datetime.fromisoformat(start_date)
    stop = datetime.fromisoformat(end_date)
    while current <= stop:
        yield current.strftime('%Y-%m-%d')
        current += timedelta(days=1)


def fetch_seatmap(
    *,
    environment: str,
    origin_location_code: str,
    destination_location_code: str,
    departure_date: str,
    travel_class: str,
    non_stop: str,
    included_airline_codes: str,
    fixtures_dir: Path = FIXTURES_DIR,
) -> SeatMap | None:
    """Retrieve seat map data for the provided criteria."""
    if environment == "e2e":
        with open(fixtures_dir / "seatmap.json", encoding="utf-8") as fixture:
            seatmap_fixtures = json.load(fixture)
        first_record = seatmap_fixtures[0] if seatmap_fixtures else None
        price_lookup = _load_flight_offer_fixtures(fixtures_dir)
        flight_offer_id = first_record.get('flightOfferId') if isinstance(first_record, dict) else None
        offer = price_lookup.get(str(flight_offer_id)) if flight_offer_id is not None else None
        price_info = offer.get('price') if isinstance(offer, dict) else None
        return _build_seatmap(first_record, price_info, captured_at=datetime.utcnow())

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
    return _build_seatmap(first_record, price_info, captured_at=datetime.utcnow())


def load_seatmaps(
    *,
    environment: str,
    travel_windows: list[dict],
    flight_search_filters: dict,
    fixtures_dir: Path = FIXTURES_DIR,
) -> list[SeatMap]:
    seatmaps: list[SeatMap] = []
    pickle_path = fixtures_dir / "seatmaps.pkl"

    if environment == "e2e-pickle":
        with open(pickle_path, "rb") as pickled_seatmaps:
            seatmaps = pickle.load(pickled_seatmaps)
        pickle_timestamp = datetime.fromtimestamp(pickle_path.stat().st_mtime)
        for seatmap_obj in seatmaps:
            if not hasattr(seatmap_obj, "price_total"):
                seatmap_obj.price_total = None
            if not hasattr(seatmap_obj, "price_currency"):
                seatmap_obj.price_currency = None
            setattr(seatmap_obj, "price_timestamp", pickle_timestamp)
        return seatmaps

    for window in travel_windows:
        for departure_date in iter_dates(window["start_date"], window["end_date"]):
            seatmap = fetch_seatmap(
                environment=environment,
                origin_location_code=window["origin"],
                destination_location_code=window["destination"],
                departure_date=departure_date,
                travel_class=flight_search_filters["travel_class"],
                non_stop=flight_search_filters["non_stop"],
                included_airline_codes=flight_search_filters["included_airline_codes"],
                fixtures_dir=fixtures_dir,
            )
            if seatmap is not None:
                seatmaps.append(seatmap)

    if seatmaps:
        with open(pickle_path, "wb") as pickled_seatmaps:
            pickle.dump(seatmaps, pickled_seatmaps)

    return seatmaps


def parse_total_price(value: str | None) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(value)
    except (InvalidOperation, ValueError):
        return None


def compute_best_price_by_route(seatmaps: Iterable[SeatMap]) -> dict[tuple[str, str], Decimal]:
    best: dict[tuple[str, str], Decimal] = {}
    for seatmap_obj in seatmaps:
        price = parse_total_price(seatmap_obj.price_total)
        if price is None:
            continue
        route = (seatmap_obj.origin, seatmap_obj.destination)
        existing = best.get(route)
        if existing is None or price < existing:
            best[route] = price
    return best


def compute_worst_price_by_route(seatmaps: Iterable[SeatMap]) -> dict[tuple[str, str], Decimal]:
    worst: dict[tuple[str, str], Decimal] = {}
    for seatmap_obj in seatmaps:
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


def build_heatmap_entries(
    seatmaps_by_date: dict[str, SeatMap],
    travel_windows: list[dict],
) -> dict[str, dict[str, Decimal]]:
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


def build_price_entries_all_dates(
    seatmaps_by_date: dict[str, SeatMap],
    travel_windows: list[dict],
) -> dict[str, dict[str, Decimal]]:
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


@dataclass(frozen=True)
class CombinationPriceRecord:
    outbound_route: str
    return_route: str
    outbound_date: str
    return_date: str
    price: Decimal
    currency: str | None
    captured_at: datetime
    outbound_window_available: bool | None = None
    return_window_available: bool | None = None


def _ensure_combination_price_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS combination_prices (
            outbound_route TEXT NOT NULL,
            return_route TEXT NOT NULL,
            outbound_date TEXT NOT NULL,
            return_date TEXT NOT NULL,
            price TEXT NOT NULL,
            currency TEXT,
            captured_at TEXT NOT NULL,
            outbound_window_available INTEGER,
            return_window_available INTEGER,
            PRIMARY KEY (outbound_route, return_route, outbound_date, return_date)
        )
        """
    )
    existing_columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(combination_prices)")
    }
    if 'outbound_window_available' not in existing_columns:
        conn.execute("ALTER TABLE combination_prices ADD COLUMN outbound_window_available INTEGER")
    if 'return_window_available' not in existing_columns:
        conn.execute("ALTER TABLE combination_prices ADD COLUMN return_window_available INTEGER")


def sync_combination_price_records(
    records: Iterable[CombinationPriceRecord],
    missing_keys: Iterable[tuple[str, str, str, str]],
    *,
    db_path: Path = ROUNDTRIP_DB_PATH,
) -> None:
    records = list(records)
    missing_keys = list(missing_keys)
    if not records and not missing_keys:
        return
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        _ensure_combination_price_table(conn)
        for outbound_route, return_route, outbound_date, return_date in missing_keys:
            conn.execute(
                """
                DELETE FROM combination_prices
                WHERE outbound_route = ?
                  AND return_route = ?
                  AND outbound_date = ?
                  AND return_date = ?
                """,
                (outbound_route, return_route, outbound_date, return_date),
            )
        for record in records:
            captured_at = record.captured_at.isoformat()
            outbound_window_available = None
            if record.outbound_window_available is not None:
                outbound_window_available = int(bool(record.outbound_window_available))
            return_window_available = None
            if record.return_window_available is not None:
                return_window_available = int(bool(record.return_window_available))
            conn.execute(
                """
                INSERT INTO combination_prices (
                    outbound_route,
                    return_route,
                    outbound_date,
                    return_date,
                    price,
                    currency,
                    captured_at,
                    outbound_window_available,
                    return_window_available
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(outbound_route, return_route, outbound_date, return_date)
                DO UPDATE SET
                    price = CASE
                        WHEN excluded.captured_at >= combination_prices.captured_at THEN excluded.price
                        ELSE combination_prices.price
                    END,
                    currency = CASE
                        WHEN excluded.captured_at >= combination_prices.captured_at THEN excluded.currency
                        ELSE combination_prices.currency
                    END,
                    captured_at = CASE
                        WHEN excluded.captured_at >= combination_prices.captured_at THEN excluded.captured_at
                        ELSE combination_prices.captured_at
                    END,
                    outbound_window_available = COALESCE(
                        excluded.outbound_window_available,
                        combination_prices.outbound_window_available
                    ),
                    return_window_available = COALESCE(
                        excluded.return_window_available,
                        combination_prices.return_window_available
                    )
                """,
                (
                    record.outbound_route,
                    record.return_route,
                    record.outbound_date,
                    record.return_date,
                    str(record.price),
                    record.currency,
                    captured_at,
                    outbound_window_available,
                    return_window_available,
                ),
            )
        conn.commit()


__all__ = [
    "SeatMap",
    "fetch_seatmap",
    "load_seatmaps",
    "iter_dates",
    "parse_total_price",
    "compute_best_price_by_route",
    "compute_worst_price_by_route",
    "has_best_price_for_route",
    "has_worst_price_for_route",
    "build_heatmap_entries",
    "build_price_entries_all_dates",
    "build_heatmap_price_stats",
    "CombinationPriceRecord",
    "sync_combination_price_records",
    "ResponseError",
]
