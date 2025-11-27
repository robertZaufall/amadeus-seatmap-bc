from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable
from urllib import request

from amadeus import Client, ResponseError
from dotenv import load_dotenv

from config import ENVIRONMENT, FLIGHT_SEARCH_FILTERS

fetch_from_api = True
refresh_data = True  # fetch even if today's data exists

data_dir = Path(__file__).parent / "data"
data_dir.mkdir(parents=True, exist_ok=True)
data_dir_today = data_dir / datetime.now().strftime("%Y%m%d")
data_dir_today.mkdir(parents=True, exist_ok=True)

FIXTURES_DIR = Path(__file__).parent / "test"

TRAVEL_DATES_FILENAME = "travel_dates.json"
SEATMAP_REQUESTS_FILENAME = "seatmap_requests.json"
SEATMAP_RESPONSES_FILENAME = "seatmap_responses.json"
SEATMAP_SUMMARY_FILENAME = "seatmaps_summary.json"
FLIGHT_OFFER_RETURN_TEMPLATE_FILENAME = "flight_offer_return_template.json"
SEATMAP_REQUEST_TEMPLATE_FILENAME = "seatmap_request_template.json"
PRICING_RESPONSES_FILENAME = "pricing_responses.json"
PRICING_RESPONSES_ONEWAY_FILENAME = "pricing_responses_oneway.json"


@dataclass
class SeatMapRecord:
    departure_date: str
    origin: str
    destination: str
    carrier: str
    number: str
    aircraft_code: str
    decks: list[Any]
    window_seats: list[str]
    price_total: str | None = None
    price_currency: str | None = None
    price_timestamp: datetime | None = None

    def to_serializable_dict(self) -> dict[str, Any]:
        return {
            "departure_date": self.departure_date,
            "origin": self.origin,
            "destination": self.destination,
            "carrier": self.carrier,
            "number": self.number,
            "aircraft_code": self.aircraft_code,
            "decks": self.decks,
            "window_seats": self.window_seats,
            "price_total": self.price_total,
            "price_currency": self.price_currency,
            "price_timestamp": self.price_timestamp.isoformat() if self.price_timestamp else None,
        }


def _dump_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def build_amadeus_client(environment: str) -> Client:
    if environment not in {"test", "production"}:
        raise ValueError("environment must be 'test' or 'production'")

    if environment == "test":
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
        hostname=environment,
    )


def get_travel_dates_tuples() -> list[list[dict[str, str]]]:
    if not (data_dir_today / TRAVEL_DATES_FILENAME).exists():
        raise RuntimeError("Travel dates data not found. Please run get_availability.py first.")
    travel_dates = _load_json(data_dir_today / TRAVEL_DATES_FILENAME)
    if not travel_dates:
        raise RuntimeError("No travel windows found in travel_dates.json.")

    max_dates = max(len(window.get("dates", [])) for window in travel_dates)
    travel_dates_tuples: list[list[dict[str, str]]] = []

    for date_idx in range(max_dates):
        entry: list[dict[str, str]] = []
        for window in travel_dates:
            dates = window.get("dates") or []
            if date_idx >= len(dates):
                continue
            entry.append(
                {
                    "origin": window["origin"],
                    "destination": window["destination"],
                    "date": dates[date_idx],
                }
            )
        if entry:
            travel_dates_tuples.append(entry)

    return travel_dates_tuples


def build_seatmap_requests(travel_dates_tuples: list[list[dict[str, str]]]) -> list[dict[str, Any]]:
    if not (data_dir_today / PRICING_RESPONSES_FILENAME).exists():
        raise RuntimeError("Pricing responses data not found. Please run get_prices.py first.")
    if not (data_dir_today / PRICING_RESPONSES_ONEWAY_FILENAME).exists():
        raise RuntimeError("Pricing responses data for oneway not found. Please run get_prices_oneway.py first.")

    pricing_payload = _load_json(data_dir_today / PRICING_RESPONSES_FILENAME)
    pricing_oneway_payload = _load_json(data_dir_today / PRICING_RESPONSES_ONEWAY_FILENAME)
 
    flight_offers = pricing_payload.get("flightOffers") if isinstance(pricing_payload, dict) else pricing_payload
    if not isinstance(flight_offers, list):
        raise RuntimeError("Unexpected pricing responses format; expected 'flightOffers' list.")

    flight_offers_oneway = pricing_oneway_payload.get("flightOffers") if isinstance(pricing_oneway_payload, dict) else pricing_oneway_payload
    if not isinstance(flight_offers_oneway, list):
        raise RuntimeError("Unexpected pricing responses format; expected 'flightOffers' list.")

    requests: list[dict[str, Any]] = []

    for element in travel_dates_tuples:
        if len(element) == 2:

            outbound, inbound = element
            origin1 = outbound["origin"]
            destination1 = outbound["destination"]
            date1 = outbound["date"]
            origin2 = inbound["origin"]
            destination2 = inbound["destination"]
            date2 = inbound["date"]

            def matches_offer(offer: dict[str, Any]) -> bool:
                itineraries = offer.get("itineraries", [])
                if len(itineraries) < 2:
                    return False

                outbound_segment = itineraries[0].get("segments", [{}])[0]
                inbound_segment = itineraries[1].get("segments", [{}])[0]

                outbound_departure = outbound_segment.get("departure", {})
                inbound_departure = inbound_segment.get("departure", {})

                outbound_date = (outbound_departure.get("at") or "").split("T")[0]
                inbound_date = (inbound_departure.get("at") or "").split("T")[0]

                return (
                    outbound_departure.get("iataCode") == origin1
                    and outbound_segment.get("arrival", {}).get("iataCode") == destination1
                    and inbound_departure.get("iataCode") == origin2
                    and inbound_segment.get("arrival", {}).get("iataCode") == destination2
                    and outbound_date == date1
                    and inbound_date == date2
                )

            flight_offer = next((offer for offer in flight_offers if matches_offer(offer)), None)
            if flight_offer:
                requests.append(flight_offer)

        else:
            single = element[0]
            origin = single["origin"]
            destination = single["destination"]
            date = single["date"]

            def matches_offer_oneway(offer: dict[str, Any]) -> bool:
                itineraries = offer.get("itineraries", [])
                if len(itineraries) < 1:
                    return False

                segment = itineraries[0].get("segments", [{}])[0]
                departure = segment.get("departure", {})

                departure_date = (departure.get("at") or "").split("T")[0]

                return (
                    departure.get("iataCode") == origin
                    and segment.get("arrival", {}).get("iataCode") == destination
                    and departure_date == date
                )

            flight_offer = next((offer for offer in flight_offers_oneway if matches_offer_oneway(offer)), None)
            if flight_offer:
                requests.append(flight_offer)

    return requests

def fetch_seatmap_batches(
    amadeus: Client, seatmap_request: Iterable[dict[str, Any]]
) -> list[Any]:
    seatmap_response = None    
    try:
        seatmap_response = amadeus.post(
            "/v1/shopping/seatmaps",
            seatmap_request,
        )
    except ResponseError as error:
        raise error
    return seatmap_response.data or []

def process_seatmaps(amadeus, travel_dates_tuples, batch_size: int = 6):
    requests_path = data_dir_today / SEATMAP_REQUESTS_FILENAME
    responses_path = data_dir_today / SEATMAP_RESPONSES_FILENAME

    if fetch_from_api:
        if refresh_data or not responses_path.exists():

            seatmap_requests = build_seatmap_requests(travel_dates_tuples)
            if not seatmap_requests:
                print("No travel windows configured; skipping seatmap fetch.")
                return
            if ENVIRONMENT in {"test", "production"} and amadeus is None:
                raise RuntimeError("Amadeus client is required to fetch seatmaps.")

            _dump_json(requests_path, seatmap_requests)

            seatmap_responses: list[Any] = []
            # process in batches
            total_requests = len(seatmap_requests)
            print(f"Total seatmap requests to process: {total_requests}")
            for idx in range(0, total_requests, batch_size):
                batch = seatmap_requests[idx : idx + batch_size]
                for i, request in enumerate(batch):
                    batch[i]["id"] = str(i)
                batch_request = {"data": batch}
                seatmap_responses.extend(fetch_seatmap_batches(amadeus, batch_request))
                print(f"Processed seatmap requests {idx + 1} to {min(idx + batch_size, total_requests)}")
            _dump_json(responses_path, seatmap_responses)


def main() -> None:
    load_dotenv()

    travel_dates_tuples = get_travel_dates_tuples()

    amadeus = build_amadeus_client(ENVIRONMENT)
    process_seatmaps(amadeus=amadeus, travel_dates_tuples=travel_dates_tuples)

if __name__ == "__main__":
    main()
