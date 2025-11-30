from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from amadeus import Client, ResponseError
from dotenv import load_dotenv

from config import ENVIRONMENT, SEATMAP_OUTPUT_STYLE
from display_utils import resolve_seatmap_style
from png_utils import save_text_block_png
from seatmap_data import SeatMap
from seatmap_display import SeatMaps


CACHE_BASE_DIR = Path(__file__).parent / "data" / "flights"
DEFAULT_CACHE_TTL_HOURS = 4.0


def build_amadeus_client(environment: str) -> Client:
    env = environment.lower()
    if env not in {"test", "production"}:
        raise RuntimeError("ENVIRONMENT must be 'test' or 'production' for live API calls.")

    if env == "test":
        client_id = os.getenv("TEST_AMADEUS_CLIENT_ID")
        client_secret = os.getenv("TEST_AMADEUS_CLIENT_SECRET")
    else:
        client_id = os.getenv("AMADEUS_CLIENT_ID")
        client_secret = os.getenv("AMADEUS_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise RuntimeError("Missing Amadeus credentials in environment variables.")

    return Client(
        client_id=client_id,
        client_secret=client_secret,
        hostname=env,
    )


def _parse_date(value: str) -> str:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date().isoformat()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Date must be in YYYY-MM-DD format.") from exc


def _parse_time(value: str) -> str:
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            parsed = datetime.strptime(value, fmt).time()
            return parsed.strftime("%H:%M:%S")
        except ValueError:
            continue
    raise argparse.ArgumentTypeError("Time must be HH:MM or HH:MM:SS.")


def _collect_available_window_seats(decks: list[dict[str, Any]] | None) -> list[str]:
    available: list[str] = []
    for deck in decks or []:
        for seat in deck.get("seats", []):
            traveler_pricing = seat.get("travelerPricing", [])
            availability = traveler_pricing[0].get("seatAvailabilityStatus") if traveler_pricing else None
            if availability != "AVAILABLE":
                continue
            if "W" not in seat.get("characteristicsCodes", []):
                continue
            seat_number = seat.get("number")
            if seat_number:
                available.append(seat_number)
    return available


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _load_json(path: Path) -> Any | None:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _sanitize_segment(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in str(value))
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or "value"


def _cache_dir_for_request(
    *,
    environment: str,
    origin: str,
    destination: str,
    date: str,
    time: str,
    travel_class: str,
    airline: str,
    currency: str,
) -> Path:
    time_squeezed = time.replace(":", "")
    parts = [
        environment.lower(),
        origin.upper(),
        destination.upper(),
        date,
        time_squeezed,
        travel_class.upper(),
        airline.upper(),
        currency.upper(),
    ]
    slug = "-".join(_sanitize_segment(part) for part in parts)
    return CACHE_BASE_DIR / slug


def _cache_paths(cache_dir: Path) -> dict[str, Path]:
    return {
        "request": cache_dir / "flight_offer_request.json",
        "offers": cache_dir / "flight_offers.json",
        "seatmaps": cache_dir / "seatmaps.json",
        "seatmaps_request": cache_dir / "seatmaps_request.json",
        "meta": cache_dir / "metadata.json",
    }


def _is_fresh(path: Path, ttl_hours: float) -> bool:
    if ttl_hours <= 0:
        return False
    if not path.exists():
        return False
    age_seconds = (datetime.now().timestamp() - path.stat().st_mtime)
    return age_seconds <= (ttl_hours * 3600)


def _load_cache(
    cache_dir: Path,
    ttl_hours: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None] | None:
    paths = _cache_paths(cache_dir)
    seatmaps_path = paths["seatmaps"]
    offers_path = paths["offers"]
    meta_path = paths["meta"]

    if not _is_fresh(seatmaps_path, ttl_hours):
        return None
    if not _is_fresh(offers_path, ttl_hours):
        return None

    seatmaps = _load_json(seatmaps_path)
    offers = _load_json(offers_path)
    meta = _load_json(meta_path)
    if not isinstance(seatmaps, list) or not isinstance(offers, list):
        return None
    return offers, seatmaps, meta


def _write_cache(
    cache_dir: Path,
    *,
    request_body: dict[str, Any],
    flight_offers: list[dict[str, Any]],
    seatmaps_request: dict[str, Any] | None = None,
    seatmaps: list[dict[str, Any]] | None = None,
) -> None:
    paths = _cache_paths(cache_dir)
    _dump_json(paths["request"], request_body)
    _dump_json(paths["offers"], flight_offers)
    if seatmaps_request is not None:
        _dump_json(paths["seatmaps_request"], seatmaps_request)
    if seatmaps is not None:
        _dump_json(paths["seatmaps"], seatmaps)
    _dump_json(
        paths["meta"],
        {"fetched_at": datetime.now().astimezone().isoformat(), "seatmaps_cached": seatmaps is not None},
    )


def build_flight_offer_request(
    *,
    origin: str,
    destination: str,
    date: str,
    time: str,
    travel_class: str,
    airline: str,
    currency: str,
    max_offers: int,
) -> dict[str, Any]:
    return {
        "currencyCode": currency.upper(),
        "originDestinations": [
            {
                "id": "1",
                "originLocationCode": origin.upper(),
                "destinationLocationCode": destination.upper(),
                "departureDateTimeRange": {"date": date, "time": time},
            }
        ],
        "travelers": [{"id": "1", "travelerType": "ADULT"}],
        "sources": ["GDS"],
        "searchCriteria": {
            "maxFlightOffers": max_offers,
            "flightFilters": {
                "cabinRestrictions": [
                    {
                        "cabin": travel_class.upper(),
                        "coverage": "MOST_SEGMENTS",
                        "originDestinationIds": ["1"],
                    }
                ],
                "carrierRestrictions": {"includedCarrierCodes": [airline.upper()]},
            },
        },
    }


def fetch_flight_offers(amadeus: Client, request_body: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        response = amadeus.post("/v2/shopping/flight-offers", request_body)
    except ResponseError as error:
        details = getattr(error, "response", None)
        message = details.body if details and hasattr(details, "body") else str(error)
        raise RuntimeError(f"Flight-offer search failed: {message}") from error

    offers = response.data or []
    if not isinstance(offers, list):
        raise RuntimeError("Unexpected flight-offer response format; expected a list.")
    return offers


def build_seatmaps_request(flight_offer: dict[str, Any]) -> dict[str, Any]:
    return {"data": [flight_offer]}


def fetch_seatmaps(amadeus: Client, seatmap_request: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        response = amadeus.post("/v1/shopping/seatmaps", seatmap_request)
    except ResponseError as error:
        details = getattr(error, "response", None)
        message = details.body if details and hasattr(details, "body") else str(error)
        raise RuntimeError(f"Seatmap fetch failed: {message}") from error

    seatmaps = response.data or []
    if not isinstance(seatmaps, list):
        raise RuntimeError("Unexpected seatmap response format; expected a list.")
    return seatmaps


def _filter_decks_for_cabin(decks: list[dict[str, Any]] | None, travel_class: str | None) -> list[dict[str, Any]]:
    if not travel_class:
        return decks or []
    cabin = travel_class.upper()
    filtered: list[dict[str, Any]] = []
    for deck in decks or []:
        seats = [seat for seat in deck.get("seats", []) if str(seat.get("cabin", "")).upper() == cabin]
        if not seats:
            continue
        deck_copy = dict(deck)
        deck_copy["seats"] = seats
        filtered.append(deck_copy)
    return filtered or (decks or [])


def build_seatmap_objects(
    records: list[dict[str, Any]],
    *,
    travel_class: str | None = None,
) -> list[tuple[SeatMap, str]]:
    mapped: list[tuple[SeatMap, str]] = []
    for record in records:
        departure_info = record.get("departure", {}) or {}
        arrival_info = record.get("arrival", {}) or {}

        departure_at = str(departure_info.get("at") or "")
        departure_iso = departure_at.split("T")[0] if "T" in departure_at else departure_at
        departure_date = departure_iso.replace("-", "") if departure_iso else ""

        decks = _filter_decks_for_cabin(record.get("decks"), travel_class)
        seatmap = SeatMap(
            departure_date=departure_date,
            origin=departure_info.get("iataCode") or "",
            destination=arrival_info.get("iataCode") or "",
            carrier=record.get("carrierCode") or "",
            number=str(record.get("number") or ""),
            aircraft_code=(record.get("aircraft") or {}).get("code") or "N/A",
            decks=decks,
            window_seats=_collect_available_window_seats(decks),
        )

        label_parts = [
            f"{seatmap.carrier}{seatmap.number}".strip(),
            f"{seatmap.origin}->{seatmap.destination}",
            departure_at or departure_iso or "",
        ]
        label = " | ".join(part for part in label_parts if part)
        mapped.append((seatmap, label))
    return mapped
def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.astimezone()
    return parsed


def _resolve_data_timestamp(meta: dict[str, Any] | None, seatmaps_path: Path | None = None) -> datetime | None:
    ts = None
    if isinstance(meta, dict):
        ts = _parse_timestamp(meta.get("fetched_at"))
    if ts:
        return ts
    if seatmaps_path and seatmaps_path.exists():
        try:
            return datetime.fromtimestamp(seatmaps_path.stat().st_mtime).astimezone()
        except OSError:
            return None
    return None


def render_seatmaps(
    entries: list[tuple[SeatMap, str]],
    *,
    image_output_path: Path | None = None,
    data_timestamp: datetime | None = None,
) -> None:
    if not entries:
        print("No seatmaps returned for this flight offer.")
        return

    seatmaps_renderer = SeatMaps([entry[0] for entry in entries])
    styles = resolve_seatmap_style(SEATMAP_OUTPUT_STYLE)
    styles = styles[:1] or ["ascii"]  # render only the first configured style (skip compact/others)
    png_lines: list[str] = []

    for style in styles:
        heading = "=== Seatmaps ===" if style == "ascii" else f"=== Seatmaps ({style}) ==="
        print(f"\n{heading}")
        print()
        if image_output_path is not None:
            if png_lines:
                png_lines.append("")
            png_lines.append(heading)
        for idx, (seatmap_obj, label) in enumerate(entries):
            label_text = label or "Seatmap"
            if idx > 0:
                print()
            print(label_text)
            rendered = seatmaps_renderer.render_map(
                seatmap_obj,
                style=style,
                thick_border=True,
                show_header=False,
            )
            cleaned = rendered.lstrip("\n")
            print(cleaned)
            if image_output_path is not None:
                if idx > 0:
                    png_lines.append("")
                png_lines.append(label_text)
                png_lines.extend(cleaned.splitlines())
                png_lines.append("")

    ts = data_timestamp or datetime.now().astimezone()
    timestamp_label = f"Data timestamp: {ts.strftime('%Y-%m-%d %H:%M:%S %Z')}"
    print()
    print(timestamp_label)

    if image_output_path is not None and png_lines:
        png_lines.append(timestamp_label)
        combined_text = "\n".join(line.rstrip() for line in png_lines).rstrip()
        save_text_block_png(
            image_output_path.stem,
            combined_text,
            output_path=image_output_path,
            occupied_replacement="XX",
        )
        print(f"\nSaved seatmap image to {image_output_path}")


def parse_args() -> argparse.Namespace:
    default_env = ENVIRONMENT if ENVIRONMENT in {"test", "production"} else "test"
    parser = argparse.ArgumentParser(
        description="Fetch seatmaps for a single flight offer and render them."
    )
    parser.add_argument("--date", required=True, type=_parse_date, help="Departure date (YYYY-MM-DD).")
    parser.add_argument(
        "--time",
        required=True,
        type=_parse_time,
        help="Departure time (HH:MM or HH:MM:SS; sent to the API as HH:MM:SS).",
    )
    parser.add_argument("--from", dest="origin", required=True, help="Origin IATA code.")
    parser.add_argument("--to", dest="destination", required=True, help="Destination IATA code.")
    parser.add_argument("--class", dest="travel_class", default="BUSINESS", help="Travel class, default BUSINESS.")
    parser.add_argument("--airline", required=True, help="Airline IATA code to filter on (e.g., TG).")
    parser.add_argument("--currency", default="EUR", help="Currency code for pricing (default: EUR).")
    parser.add_argument(
        "--max-offers",
        type=int,
        default=1,
        help="Maximum flight offers to request (seatmaps fetched only when exactly one is returned).",
    )
    parser.add_argument(
        "--environment",
        default=default_env,
        choices=["test", "production"],
        help=f"Amadeus host to use (default: {default_env}).",
    )
    parser.add_argument(
        "--cache-ttl-hours",
        type=float,
        default=DEFAULT_CACHE_TTL_HOURS,
        help="Max age for cached responses before refetching (hours). Use 0 to force refresh.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore cached data and fetch fresh responses.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    CACHE_BASE_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = _cache_dir_for_request(
        environment=args.environment,
        origin=args.origin,
        destination=args.destination,
        date=args.date,
        time=args.time,
        travel_class=args.travel_class,
        airline=args.airline,
        currency=args.currency,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_paths = _cache_paths(cache_dir)
    seatmaps_request_path = cache_paths["seatmaps_request"]

    seatmaps_request: dict[str, Any] | None = _load_json(seatmaps_request_path)
    seatmap_records: list[dict[str, Any]] = []
    cache_meta: dict[str, Any] | None = None
    cached_offers = _load_json(cache_paths["offers"])
    flight_offers: list[dict[str, Any]] = cached_offers if isinstance(cached_offers, list) else []

    if isinstance(seatmaps_request, dict):
        cached_seatmaps = _load_json(cache_paths["seatmaps"])
        cache_meta = _load_json(cache_paths["meta"])
        use_cached_seatmaps = (
            not args.force_refresh
            and _is_fresh(cache_paths["seatmaps"], args.cache_ttl_hours)
            and isinstance(cached_seatmaps, list)
        )
        if use_cached_seatmaps:
            print(f"Using cached seatmaps from {cache_dir}.")
            seatmap_records = cached_seatmaps  # type: ignore[assignment]
        else:
            print("Refreshing seatmaps using existing seatmaps_request.json (skipping flight-offer search).")
            amadeus = build_amadeus_client(args.environment)
            seatmap_records = fetch_seatmaps(amadeus, seatmaps_request)
            _dump_json(cache_paths["seatmaps"], seatmap_records)
            cache_meta = {"fetched_at": datetime.now().astimezone().isoformat(), "seatmaps_cached": True}
            _dump_json(cache_paths["meta"], cache_meta)
    else:
        cached_payload = None if args.force_refresh else _load_cache(cache_dir, args.cache_ttl_hours)
        if cached_payload:
            print(f"Using cached data from {cache_dir}.")
            flight_offers, seatmap_records, cache_meta = cached_payload
        else:
            amadeus = build_amadeus_client(args.environment)
            flight_offer_request = build_flight_offer_request(
                origin=args.origin,
                destination=args.destination,
                date=args.date,
                time=args.time,
                travel_class=args.travel_class,
                airline=args.airline,
                currency=args.currency,
                max_offers=args.max_offers,
            )

            flight_offers = fetch_flight_offers(amadeus, flight_offer_request)
            _write_cache(cache_dir, request_body=flight_offer_request, flight_offers=flight_offers, seatmaps=None)

            offer_count = len(flight_offers)
            print(f"Flight-offer search returned {offer_count} result(s).")

            if offer_count != 1:
                print("Seatmaps are only fetched when the search returns exactly one offer.")
                return

            seatmaps_request = build_seatmaps_request(flight_offers[0])
            seatmap_records = fetch_seatmaps(amadeus, seatmaps_request)
            _write_cache(
                cache_dir,
                request_body=flight_offer_request,
                flight_offers=flight_offers,
                seatmaps_request=seatmaps_request,
                seatmaps=seatmap_records,
            )
            cache_meta = {"fetched_at": datetime.now().astimezone().isoformat(), "seatmaps_cached": True}

    offer_count = len(flight_offers)
    if isinstance(seatmaps_request, dict) and offer_count == 0:
        data = seatmaps_request.get("data")
        if isinstance(data, list):
            offer_count = len(data)
    print(f"Flight-offer search returned {offer_count} result(s).")
    if seatmaps_request is None and offer_count != 1:
        print("Seatmaps are only fetched when the search returns exactly one offer.")
        return

    if seatmaps_request is None and flight_offers and len(flight_offers) == 1 and not seatmaps_request_path.exists():
        seatmaps_request = build_seatmaps_request(flight_offers[0])
        _dump_json(seatmaps_request_path, seatmaps_request)

    render_seatmaps(
        build_seatmap_objects(seatmap_records, travel_class=args.travel_class),
        image_output_path=cache_dir / "seatmaps.png",
        data_timestamp=_resolve_data_timestamp(cache_meta, cache_paths.get("seatmaps")),
    )


if __name__ == "__main__":
    main()
