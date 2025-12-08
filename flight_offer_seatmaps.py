from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from amadeus import Client, ResponseError
from dotenv import load_dotenv

from config import ENVIRONMENT, SEATMAP_OUTPUT_STYLE
from display_utils import extract_row_and_column, resolve_seatmap_style
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


def _hash_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _build_metadata(
    seatmaps_cached: bool,
    seatmaps_path: Path,
    existing_meta: dict[str, Any] | None = None,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    existing_meta = existing_meta if isinstance(existing_meta, dict) else {}
    seatmaps_hash = existing_meta.get("seatmaps_hash")
    previous_hash = existing_meta.get("previous_seatmaps_hash")
    stored_args = existing_meta.get("command_args")

    if seatmaps_cached and seatmaps_path.exists():
        new_hash = _hash_file(seatmaps_path)
        if new_hash == seatmaps_hash:
            # If unchanged across runs, keep backup aligned with the current hash.
            previous_hash = new_hash
        else:
            # When the hash changes, shift the current hash into the backup slot.
            previous_hash = seatmaps_hash
            seatmaps_hash = new_hash
    if command_args is None:
        command_args = stored_args
    return {
        "fetched_at": datetime.now().astimezone().isoformat(),
        "seatmaps_cached": seatmaps_cached,
        "seatmaps_hash": seatmaps_hash,
        "previous_seatmaps_hash": previous_hash,
        "command_args": command_args,
    }


def _current_command_args() -> list[str]:
    """Return the current CLI arguments (excluding the program name)."""
    return sys.argv[1:]


def _strip_medias(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_medias(v) for k, v in value.items() if k != "medias"}
    if isinstance(value, list):
        return [_strip_medias(item) for item in value]
    return value


def _sanitize_seatmaps(seatmaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_strip_medias(entry) if isinstance(entry, dict) else entry for entry in seatmaps]


def _write_seatmaps_with_diff(path: Path, seatmaps: list[dict[str, Any]]) -> Path | None:
    seatmaps = _sanitize_seatmaps(seatmaps)
    diff_path = path.with_suffix(".diff")
    try:
        previous_text = path.read_text(encoding="utf-8")
    except OSError:
        previous_text = None

    new_text = json.dumps(seatmaps, indent=2)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new_text, encoding="utf-8")

    if previous_text is None:
        if diff_path.exists():
            diff_path.unlink(missing_ok=True)
        return None

    if previous_text == new_text:
        if diff_path.exists():
            diff_path.unlink(missing_ok=True)
        return None

    diff_lines = difflib.unified_diff(
        previous_text.splitlines(),
        new_text.splitlines(),
        fromfile=f"{path.name} (previous)",
        tofile=f"{path.name} (new)",
        lineterm="",
    )
    diff_text = "\n".join(diff_lines)
    if not diff_text.strip():
        if diff_path.exists():
            diff_path.unlink(missing_ok=True)
        return None

    diff_path.write_text(diff_text + "\n", encoding="utf-8")
    return diff_path


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
    travel_class: str | None,
    airline: str,
    currency: str,
) -> Path:
    time_squeezed = time.replace(":", "")
    travel_class_label = (travel_class or "MULTI").upper()
    parts = [
        environment.lower(),
        origin.upper(),
        destination.upper(),
        date,
        time_squeezed,
        travel_class_label,
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


def _is_fresh(path: Path, ttl_hours: float, *, fetched_at: str | None = None) -> bool:
    """Return True when both the file and metadata timestamps are within the TTL window."""
    if ttl_hours <= 0:
        return False
    if not path.exists():
        return False

    timestamps: list[float] = []
    if fetched_at:
        parsed = _parse_timestamp(fetched_at)
        if parsed:
            timestamps.append(parsed.timestamp())
    try:
        timestamps.append(path.stat().st_mtime)
    except OSError:
        pass
    if not timestamps:
        return False

    age_seconds = datetime.now().timestamp() - max(timestamps)
    return age_seconds <= (ttl_hours * 3600)


def _load_cache(
    cache_dir: Path,
    ttl_hours: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None] | None:
    paths = _cache_paths(cache_dir)
    seatmaps_path = paths["seatmaps"]
    offers_path = paths["offers"]
    meta_path = paths["meta"]

    meta = _load_json(meta_path)
    fetched_at = meta.get("fetched_at") if isinstance(meta, dict) else None

    if not _is_fresh(seatmaps_path, ttl_hours, fetched_at=fetched_at):
        return None
    if not _is_fresh(offers_path, ttl_hours, fetched_at=fetched_at):
        return None

    seatmaps = _load_json(seatmaps_path)
    offers = _load_json(offers_path)
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
) -> dict[str, Any]:
    paths = _cache_paths(cache_dir)
    existing_meta = _load_json(paths["meta"])
    _dump_json(paths["request"], request_body)
    _dump_json(paths["offers"], flight_offers)
    if seatmaps_request is not None:
        _dump_json(paths["seatmaps_request"], seatmaps_request)
    if seatmaps is not None:
        _write_seatmaps_with_diff(paths["seatmaps"], seatmaps)
    metadata = _build_metadata(
        seatmaps is not None,
        paths["seatmaps"],
        existing_meta,
        _current_command_args(),
    )
    _dump_json(paths["meta"], metadata)
    return metadata


def build_flight_offer_request(
    *,
    origin: str,
    destination: str,
    date: str,
    time: str,
    travel_class: str | list[str] | None,
    airline: str,
    currency: str,
    max_offers: int,
) -> dict[str, Any]:
    cabin_restrictions: list[dict[str, Any]] | None
    if travel_class is None:
        cabin_restrictions = None  # allow all cabins in a single request
    elif isinstance(travel_class, list):
        cabins = [c.upper() for c in travel_class if c]
        cabin_restrictions = [
            {
                "cabin": cabin,
                "coverage": "MOST_SEGMENTS",
                "originDestinationIds": ["1"],
            }
            for cabin in cabins
        ]
    else:
        cabin_restrictions = [
            {
                "cabin": travel_class.upper(),
                "coverage": "MOST_SEGMENTS",
                "originDestinationIds": ["1"],
            }
        ]

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
                **(
                    {"cabinRestrictions": cabin_restrictions}
                    if cabin_restrictions is not None
                    else {}
                ),
                "carrierRestrictions": {"includedCarrierCodes": [airline.upper()]},
            },
        },
    }


def _post_flight_offer_search(amadeus: Client, payload: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        response = amadeus.post("/v2/shopping/flight-offers", payload)
    except ResponseError as error:
        details = getattr(error, "response", None)
        message = details.body if details and hasattr(details, "body") else str(error)
        raise RuntimeError(f"Flight-offer search failed: {message}") from error

    offers = response.data or []
    if isinstance(offers, list) and offers and all(isinstance(item, dict) and "data" in item for item in offers):
        flattened: list[dict[str, Any]] = []
        for entry in offers:
            entry_data = entry.get("data")
            if isinstance(entry_data, list):
                flattened.extend(entry_data)
        offers = flattened
    elif isinstance(offers, list) and offers and all(isinstance(item, list) for item in offers):
        flattened = []
        for entry in offers:
            flattened.extend(entry)
        offers = flattened
    if not isinstance(offers, list):
        raise RuntimeError("Unexpected flight-offer response format; expected a list.")
    return offers


def fetch_flight_offers(amadeus: Client, request_body: dict[str, Any]) -> list[dict[str, Any]]:
    batched = request_body.get("requests")
    if isinstance(batched, list):
        combined: list[dict[str, Any]] = []
        for idx, payload in enumerate(batched, start=1):
            if not isinstance(payload, dict):
                continue
            print(f"Fetching flight offers for batched query {idx}/{len(batched)}.")
            source_cabin = _payload_cabin(payload)
            offers = _post_flight_offer_search(amadeus, payload)
            for offer in offers:
                offer_copy = json.loads(json.dumps(offer))
                if source_cabin:
                    offer_copy["__source_cabin"] = source_cabin
                combined.append(offer_copy)
        return combined
    return _post_flight_offer_search(amadeus, request_body)


def build_seatmaps_request(flight_offers: list[dict[str, Any]]) -> dict[str, Any]:
    return {"data": flight_offers}


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
) -> list[tuple[SeatMap, str, str]]:
    mapped: list[tuple[SeatMap, str, str]] = []
    for record in records:
        departure_info = record.get("departure", {}) or {}
        arrival_info = record.get("arrival", {}) or {}

        departure_at = str(departure_info.get("at") or "")
        departure_iso = departure_at.split("T")[0] if "T" in departure_at else departure_at
        departure_date = departure_iso.replace("-", "") if departure_iso else ""
        departure_label = _format_departure_label(departure_at)

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
            _cabin_label_from_seatmap_record(record, travel_class),
            f"{seatmap.origin}->{seatmap.destination}",
            departure_label or departure_at or departure_iso or "",
        ]
        label = " | ".join(part for part in label_parts if part)
        mapped.append((seatmap, label, departure_at))
    return mapped


def _format_departure_label(value: str | None) -> str:
    """Format an ISO-like departure datetime as 'YYYY-MM-DD HH:MM'."""
    if not value:
        return ""
    try:
        dt = datetime.fromisoformat(value)
        return dt.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        if "T" in value:
            date_part, time_part = value.split("T", 1)
            trimmed_time = time_part.split(":")
            if len(trimmed_time) >= 2:
                return f"{date_part} {trimmed_time[0]}:{trimmed_time[1]}"
            return f"{date_part} {time_part}"
        return value


def _extract_offer_cabin(flight_offer: dict[str, Any]) -> str | None:
    """Extract the cabin (BUSINESS/ECONOMY/ETC) from the first segment pricing."""
    pr = flight_offer.get("travelerPricings") or []
    if not pr:
        return None
    fare_details = pr[0].get("fareDetailsBySegment") or []
    if not fare_details:
        return None
    cabin = fare_details[0].get("cabin")
    return str(cabin).upper() if cabin else None


def _select_offers_for_cabins(
    flight_offers: list[dict[str, Any]],
    cabins: list[str],
) -> list[dict[str, Any]]:
    """Pick the best matching offer for each requested cabin, preferring offers sourced from that cabin request."""
    def itinerary_signature(offer: dict[str, Any]) -> list[tuple[str, str]]:
        sig: list[tuple[str, str]] = []
        for itinerary in offer.get("itineraries") or []:
            for seg in itinerary.get("segments") or []:
                carrier = str(seg.get("carrierCode") or "").upper()
                number = str(seg.get("number") or "")
                sig.append((carrier, number))
        return sig

    def score_against_anchor(candidate: dict[str, Any], anchor: dict[str, Any]) -> tuple[int, int]:
        cand_sig = itinerary_signature(candidate)
        anchor_sig = itinerary_signature(anchor)
        # Score by positional matches first, then total overlap size, then shorter length diff.
        positional = sum(1 for a, b in zip(cand_sig, anchor_sig) if a == b)
        overlap = len(set(cand_sig) & set(anchor_sig))
        length_penalty = abs(len(cand_sig) - len(anchor_sig))
        return (positional, overlap, -length_penalty)

    remaining = [c.upper() for c in cabins]
    selected: list[dict[str, Any]] = []
    anchor_offer: dict[str, Any] | None = None
    for cabin in remaining:
        # Prefer an offer tagged with the matching source cabin
        tagged_candidates = [offer for offer in flight_offers if offer.get("__source_cabin") == cabin]
        fallback_candidates = [offer for offer in flight_offers if _extract_offer_cabin(offer) == cabin]

        candidates = tagged_candidates or fallback_candidates
        if not candidates:
            continue

        if anchor_offer is not None:
            best = max(candidates, key=lambda cand: score_against_anchor(cand, anchor_offer))
        else:
            best = candidates[0]

        selected.append(best)
        if anchor_offer is None:
            anchor_offer = best
    return selected


def _payload_cabin(payload: dict[str, Any]) -> str | None:
    """Return the cabin restriction from a flight-offer request payload, if present."""
    try:
        restrictions = payload["searchCriteria"]["flightFilters"]["cabinRestrictions"]
        if isinstance(restrictions, list) and restrictions:
            cabin = restrictions[0].get("cabin")
            return str(cabin).upper() if cabin else None
    except Exception:
        return None
    return None


def _cabin_label_from_seatmap_record(record: dict[str, Any], travel_class: str | None) -> str:
    """Derive a displayable cabin label from seat entries or fallback to requested class."""
    cabins: set[str] = set()
    for deck in record.get("decks") or []:
        for seat in deck.get("seats", []) or []:
            cabin = seat.get("cabin")
            if cabin:
                cabins.add(str(cabin).upper())
    if cabins:
        return "/".join(sorted(c.title() for c in cabins))
    if travel_class:
        return travel_class.title()
    return ""


def _dedupe_offer_ids_for_seatmaps(offers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return deep-copied offers with unique IDs to satisfy seatmap API requirements."""
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for offer in offers:
        copy_offer = json.loads(json.dumps(offer))
        copy_offer.pop("__source_cabin", None)
        base_id = str(copy_offer.get("id") or "offer")
        unique_id = base_id
        counter = 1
        while unique_id in seen:
            counter += 1
            unique_id = f"{base_id}-{counter}"
        copy_offer["id"] = unique_id
        seen.add(unique_id)
        deduped.append(copy_offer)
    return deduped


def _extract_row_number(seat: dict[str, Any]) -> int | None:
    """Best-effort extraction of the numeric row from a seat entry."""
    seat_number = seat.get("number")
    row_str, _ = extract_row_and_column(str(seat_number or ""))
    if row_str.isdigit():
        return int(row_str)

    coords = seat.get("coordinates") or {}
    for key in ("x", "row", "rowNumber"):
        value = coords.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
    return None


def _seatmap_min_row(seatmap: SeatMap) -> int:
    """Return the lowest numeric row present in a seatmap (used for consistent ordering)."""
    min_row: int | None = None
    for deck in seatmap.decks or []:
        for seat in deck.get("seats", []):
            row_value = _extract_row_number(seat)
            if row_value is None:
                continue
            min_row = row_value if min_row is None else min(min_row, row_value)
    return min_row if min_row is not None else 10**9


def _seatmap_row_bounds(seatmap: SeatMap) -> tuple[int, int]:
    """Return (min_row, deck_hint) for sorting seatmaps that share the same flight."""
    min_row: int | None = None
    deck_hint: int | None = None
    for deck in seatmap.decks or []:
        if deck_hint is None:
            try:
                deck_hint = int(deck.get("deckNumber", 0) or 0)
            except (TypeError, ValueError):
                deck_hint = 0
        for seat in deck.get("seats", []):
            row_value = _extract_row_number(seat)
            if row_value is None:
                continue
            min_row = row_value if min_row is None else min(min_row, row_value)
    return (min_row if min_row is not None else 10**9, deck_hint if deck_hint is not None else 10**6)


def _seatmap_departure_sort_key(seatmap: SeatMap, departure_at: str | None) -> tuple[int, Any]:
    """Sort primarily by departure timestamp, then fall back to date-only or empty."""
    parsed = _parse_timestamp(departure_at)
    if parsed is not None:
        return (0, parsed)
    if seatmap.departure_date:
        return (1, seatmap.departure_date)
    return (2, "")


def _seatmap_cabin_priority(seatmap: SeatMap) -> int:
    """Return a priority value to render business-class layouts before economy when grouped."""
    cabins: set[str] = set()
    for deck in seatmap.decks or []:
        for seat in deck.get("seats", []) or []:
            cabin = seat.get("cabin")
            if cabin:
                cabins.add(str(cabin).upper())
    if "BUSINESS" in cabins:
        return 0
    return 1
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
    entries: list[tuple[SeatMap, str, str]],
    *,
    image_output_path: Path | None = None,
    data_timestamp: datetime | None = None,
) -> None:
    if not entries:
        print("No seatmaps returned for this flight offer.")
        return

    ordered_entries = sorted(
        entries,
        key=lambda item: (
            _seatmap_departure_sort_key(item[0], item[2]),
            _seatmap_cabin_priority(item[0]),
            _seatmap_row_bounds(item[0]),
            item[1],
        ),
    )
    seatmaps_renderer = SeatMaps([entry[0] for entry in ordered_entries])
    styles = resolve_seatmap_style(SEATMAP_OUTPUT_STYLE)
    styles = styles[:1] or ["ascii"]  # render only the first configured style (skip compact/others)
    png_lines: list[str] = []

    def _pad_to_width(text: str, width: int) -> str:
        from display_utils import display_width as _dw

        pad = max(width - _dw(text), 0)
        return text + (" " * pad)

    def _collapse_group(group_entries: list[tuple[str, list[str]]]) -> list[str]:
        lines: list[str] = []
        for idx, (label, content) in enumerate(group_entries):
            if idx > 0:
                lines.append("")
            parts = label.split(" | ")
            if parts:
                title = " | ".join(parts[:-1]) if len(parts) > 1 else parts[0]
                subtitle = parts[-1] if len(parts) > 1 else ""
                lines.append(title)
                if subtitle:
                    lines.append(subtitle)
            else:
                lines.append(label)
            lines.extend(content)
        return lines

    def _combine_columns(left: list[str], right: list[str]) -> list[str]:
        if not right:
            return left
        from display_utils import display_width as _dw

        left_width = max((_dw(line) for line in left), default=0)
        combined: list[str] = []
        max_len = max(len(left), len(right))
        for idx in range(max_len):
            l_text = left[idx] if idx < len(left) else ""
            r_text = right[idx] if idx < len(right) else ""
            combined.append(f"{_pad_to_width(l_text, left_width)}    {r_text}")
        return combined

    def _group_entries(rendered: list[tuple[SeatMap, str, list[str], str]]) -> list[list[tuple[str, list[str]]]]:
        """Group seatmaps into pairs (Business/Economy) per flight, preserving order."""
        if not rendered:
            return []
        groups: list[list[tuple[str, list[str]]]] = []
        for idx in range(0, len(rendered), 2):
            chunk = rendered[idx : idx + 2]
            groups.append([(label, lines) for _, label, lines, _ in chunk])
        return groups

    def _pad_group_to_align(groups: list[list[tuple[str, list[str]]]]) -> list[list[str]]:
        """Pad business blocks so economies start aligned across columns."""
        padded_columns: list[list[str]] = []
        max_business_height = 0
        business_blocks: list[list[str]] = []
        economy_blocks: list[list[str]] = []
        for group in groups:
            biz_lines = _collapse_group(group[:1]) if group else []
            econ_lines = _collapse_group(group[1:2]) if len(group) > 1 else []
            business_blocks.append(biz_lines)
            economy_blocks.append(econ_lines)
            max_business_height = max(max_business_height, len(biz_lines))
        for biz, econ in zip(business_blocks, economy_blocks):
            padded = list(biz)
            if len(padded) < max_business_height:
                padded.extend([""] * (max_business_height - len(padded)))
            if econ:
                if padded:
                    padded.append("")
                padded.extend(econ)
            padded_columns.append(padded)
        return padded_columns

    def _render_style(style: str) -> list[str]:
        rendered_entries: list[tuple[SeatMap, str, list[str], str]] = []
        for seatmap_obj, label, departure_at in ordered_entries:
            rendered = seatmaps_renderer.render_map(
                seatmap_obj,
                style=style,
                thick_border=True,
                show_header=False,
            )
            cleaned = rendered.lstrip("\n")
            rendered_entries.append((seatmap_obj, label, cleaned.splitlines(), departure_at))

        grouped = _group_entries(rendered_entries)
        grouped = _pad_group_to_align(grouped)
        blocks: list[str] = []
        for idx in range(0, len(grouped), 2):
            left_group = grouped[idx]
            right_group = grouped[idx + 1] if idx + 1 < len(grouped) else None
            left_lines = left_group
            right_lines = right_group if right_group else []
            combined = _combine_columns(left_lines, right_lines)
            if blocks and combined:
                blocks.append("")
            blocks.extend(combined)
        return blocks

    for style in styles:
        lines = _render_style(style)
        if lines:
            for line in lines:
                print(line)
            if image_output_path is not None:
                if png_lines:
                    png_lines.append("")
                png_lines.extend(lines)
                png_lines.append("")

    if image_output_path is not None and png_lines:
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
    parser.add_argument(
        "--class",
        dest="travel_class",
        default=None,
        help="Travel class (omit to query both BUSINESS and ECONOMY).",
    )
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

    travel_class = args.travel_class
    cabins = [travel_class] if travel_class else ["BUSINESS", "ECONOMY"]
    expected_seatmaps = len(cabins)
    CACHE_BASE_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = _cache_dir_for_request(
        environment=args.environment,
        origin=args.origin,
        destination=args.destination,
        date=args.date,
        time=args.time,
        travel_class=travel_class,
        airline=args.airline,
        currency=args.currency,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_paths = _cache_paths(cache_dir)
    seatmaps_request_path = cache_paths["seatmaps_request"]

    raw_seatmaps_request: dict[str, Any] | None = _load_json(seatmaps_request_path)
    seatmaps_request: dict[str, Any] | None = None
    if isinstance(raw_seatmaps_request, dict):
        data_entries = raw_seatmaps_request.get("data")
        if isinstance(data_entries, list):
            # Always honor an existing seatmaps_request.json, deduping IDs as needed.
            seatmaps_request = build_seatmaps_request(_dedupe_offer_ids_for_seatmaps(data_entries))
        else:
            seatmaps_request = raw_seatmaps_request
    seatmap_records: list[dict[str, Any]] = []
    cache_meta: dict[str, Any] | None = None
    cached_offers = _load_json(cache_paths["offers"])
    flight_offers: list[dict[str, Any]] = cached_offers if isinstance(cached_offers, list) else []

    if isinstance(seatmaps_request, dict):
        cached_seatmaps = _load_json(cache_paths["seatmaps"])
        cache_meta = _load_json(cache_paths["meta"])
        fetched_at = cache_meta.get("fetched_at") if isinstance(cache_meta, dict) else None
        use_cached_seatmaps = (
            not args.force_refresh
            and _is_fresh(cache_paths["seatmaps"], args.cache_ttl_hours, fetched_at=fetched_at)
            and isinstance(cached_seatmaps, list)
            and len(cached_seatmaps) >= expected_seatmaps
        )
        if use_cached_seatmaps:
            print(f"Using cached seatmaps from {cache_dir}.")
            seatmap_records = cached_seatmaps  # type: ignore[assignment]
            cache_meta = _build_metadata(True, cache_paths["seatmaps"], cache_meta, _current_command_args())
            _dump_json(cache_paths["meta"], cache_meta)
        else:
            print("Refreshing seatmaps using existing seatmaps_request.json (skipping flight-offer search).")
            amadeus = build_amadeus_client(args.environment)
            seatmap_records = _sanitize_seatmaps(fetch_seatmaps(amadeus, seatmaps_request))
            _write_seatmaps_with_diff(cache_paths["seatmaps"], seatmap_records)
            existing_meta = _load_json(cache_paths["meta"])
            cache_meta = _build_metadata(True, cache_paths["seatmaps"], existing_meta, _current_command_args())
            _dump_json(cache_paths["meta"], cache_meta)
    else:
        cached_payload = None if args.force_refresh else _load_cache(cache_dir, args.cache_ttl_hours)
        if cached_payload:
            print(f"Using cached data from {cache_dir}.")
            flight_offers, seatmap_records, cache_meta = cached_payload
            if len(seatmap_records) < expected_seatmaps:
                print("Cached seatmaps incomplete for requested cabins; refreshing.")
                cached_payload = None
                seatmap_records = []
                flight_offers = []
            else:
                cache_meta = _build_metadata(True, cache_paths["seatmaps"], cache_meta, _current_command_args())
                _dump_json(cache_paths["meta"], cache_meta)
        if not cached_payload:
            amadeus = build_amadeus_client(args.environment)
            if travel_class is None:
                batched_requests = [
                    build_flight_offer_request(
                        origin=args.origin,
                        destination=args.destination,
                        date=args.date,
                        time=args.time,
                        travel_class=cabin,
                        airline=args.airline,
                        currency=args.currency,
                        max_offers=(2 if cabin.upper() == "ECONOMY" else args.max_offers),
                    )
                    for cabin in cabins
                ]
                flight_offer_request: dict[str, Any] = {"requests": batched_requests}
                for cabin, payload in zip(cabins, batched_requests):
                    out_path = cache_dir / f"flight_offer_request_{cabin.lower()}.json"
                    _dump_json(out_path, payload)
            else:
                flight_offer_request = build_flight_offer_request(
                    origin=args.origin,
                    destination=args.destination,
                    date=args.date,
                    time=args.time,
                    travel_class=travel_class,
                    airline=args.airline,
                    currency=args.currency,
                    max_offers=args.max_offers,
                )

            flight_offers = fetch_flight_offers(amadeus, flight_offer_request)
            _write_cache(cache_dir, request_body=flight_offer_request, flight_offers=flight_offers, seatmaps=None)

            offer_count = len(flight_offers)
            if travel_class is None:
                print(f"Flight-offer search returned {offer_count} result(s) across cabins {', '.join(cabins)}.")
                selected_offers = _select_offers_for_cabins(flight_offers, cabins)
                if not selected_offers:
                    print("No matching offers found for requested cabins; skipping seatmaps.")
                    return
                seatmaps_request = build_seatmaps_request(_dedupe_offer_ids_for_seatmaps(selected_offers))
            else:
                print(f"Flight-offer search returned {offer_count} result(s).")
                if offer_count != 1:
                    print("Seatmaps are only fetched when the search returns exactly one offer.")
                    return
                seatmaps_request = build_seatmaps_request(_dedupe_offer_ids_for_seatmaps([flight_offers[0]]))

            seatmap_records = _sanitize_seatmaps(fetch_seatmaps(amadeus, seatmaps_request))
            cache_meta = _write_cache(
                cache_dir,
                request_body=flight_offer_request,
                flight_offers=flight_offers,
                seatmaps_request=seatmaps_request,
                seatmaps=seatmap_records,
            )

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
        seatmaps_request = build_seatmaps_request(_dedupe_offer_ids_for_seatmaps([flight_offers[0]]))
        _dump_json(seatmaps_request_path, seatmaps_request)

    render_seatmaps(
        build_seatmap_objects(seatmap_records, travel_class=travel_class),
        image_output_path=cache_dir / "seatmaps.png",
        data_timestamp=_resolve_data_timestamp(cache_meta, cache_paths.get("seatmaps")),
    )


if __name__ == "__main__":
    main()
