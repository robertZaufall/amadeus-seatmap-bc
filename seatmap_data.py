from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Iterable, Iterator

from config import CURRENCY_SYMBOLS


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


def iter_dates(start_date: str, end_date: str) -> Iterator[str]:
    current = datetime.fromisoformat(start_date)
    stop = datetime.fromisoformat(end_date)
    while current <= stop:
        yield current.strftime('%Y-%m-%d')
        current += timedelta(days=1)


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


def extract_layout_snapshot(seatmap_obj: SeatMap) -> list[dict]:
    """Return a pared-down representation of the seat layout suitable for storage."""
    layout: list[dict] = []
    for deck in seatmap_obj.decks or []:
        deck_snapshot = {
            "deckType": deck.get("deckType"),
            "deckNumber": deck.get("deckNumber"),
            "deckConfiguration": deck.get("deckConfiguration"),
            "seats": [],
        }
        for seat in deck.get("seats", []):
            traveler_pricing = seat.get("travelerPricing", [])
            if traveler_pricing:
                pricing_snapshot = [{
                    "seatAvailabilityStatus": traveler_pricing[0].get("seatAvailabilityStatus")
                }]
            else:
                pricing_snapshot = []
            deck_snapshot["seats"].append(
                {
                    "number": seat.get("number"),
                    "coordinates": seat.get("coordinates"),
                    "travelerPricing": pricing_snapshot,
                    "characteristicsCodes": seat.get("characteristicsCodes", []),
                }
            )
        layout.append(deck_snapshot)
    return layout


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


__all__ = [
    "SeatMap",
    "iter_dates",
    "parse_total_price",
    "compute_best_price_by_route",
    "compute_worst_price_by_route",
    "has_best_price_for_route",
    "has_worst_price_for_route",
    "extract_layout_snapshot",
    "build_heatmap_entries",
    "build_price_entries_all_dates",
    "build_heatmap_price_stats",
]
