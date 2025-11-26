from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

from amadeus import Client, ResponseError
from dotenv import load_dotenv

from config import TRAVEL_WINDOWS, FLIGHT_SEARCH_FILTERS, ENVIRONMENT

fetch_from_api = True
refresh_data = True # get data even data is present from today

data_dir = Path(__file__).parent / "data"
data_dir.mkdir(parents=True, exist_ok=True)
data_dir_today = data_dir / datetime.now().strftime("%Y%m%d")
data_dir_today.mkdir(parents=True, exist_ok=True)


AVAILABILITY_REQUESTS_FILENAME = "availability_requests.json"
AVAILABILITY_RESPONSES_FILENAME = "availability_responses.json"
UNAVAILABLE_FLIGHTS_FILENAME = "unavailable_flights.json"
TRAVEL_DATES_FILENAME = "travel_dates.json"


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


def _build_request_template() -> dict[str, Any]:
    max_number_of_connections = 0 if FLIGHT_SEARCH_FILTERS["non_stop"] == "true" else 1
    return {
        "originDestinations": [],
        "travelers": [{"id": "1", "travelerType": "ADULT"}],
        "sources": ["GDS"],
        "searchCriteria": {
            "maxFlightOffers": 1,
            "flightFilters": {
                "cabinRestrictions": [
                    {
                        "cabin": FLIGHT_SEARCH_FILTERS["travel_class"],
                        "coverage": "MOST_SEGMENTS",
                        "originDestinationIds": [],
                    }
                ],
                "carrierRestrictions": {
                    "includedCarrierCodes": [FLIGHT_SEARCH_FILTERS["included_airline_codes"]]
                },
                "connectionRestriction": {"maxNumberOfConnections": max_number_of_connections},
            },
        },
    }


def build_availability_requests(
    travel_dates: Iterable[dict[str, Any]],
    *,
    batch_size: int = 6,
) -> list[dict[str, Any]]:
    request_template = _build_request_template()
    availability_requests: list[dict[str, Any]] = []
    availability_request = deepcopy(request_template)

    for travel_date in travel_dates:
        dates = travel_date.get("dates", [])
        for date in dates:
            if len(availability_request["originDestinations"]) == batch_size:
                availability_requests.append(availability_request)
                availability_request = deepcopy(request_template)

            availability_request["originDestinations"].append(
                {
                    "id": str(len(availability_request["originDestinations"]) + 1),
                    "originLocationCode": travel_date["origin"],
                    "destinationLocationCode": travel_date["destination"],
                    "departureDateTime": {"date": date},
                }
            )
            availability_request["searchCriteria"]["flightFilters"]["cabinRestrictions"][0][
                "originDestinationIds"
            ].append(str(len(availability_request["originDestinations"])))

    if availability_request["originDestinations"]:
        availability_requests.append(availability_request)

    return availability_requests


def fetch_availability_batches(
    amadeus: Client, availability_requests: Iterable[dict[str, Any]]
) -> list[Any]:
    availability_responses: list[Any] = []
    for idx, request in enumerate(availability_requests, start=1):
        print(f"Availability request batch {idx}:")
        try:
            availability_response = amadeus.post(
                "/v1/shopping/availability/flight-availabilities",
                request,
            )
            availability_responses.append(availability_response.data)
        except ResponseError as error:
            raise error
        print(f"Availability request batch {idx} done.")
    return availability_responses


def compute_unavailable_flights(
    availability_requests: Iterable[dict[str, Any]],
    availability_responses: Iterable[Iterable[dict[str, Any]]],
) -> list[dict[str, str]]:
    unavailable_flights: list[dict[str, str]] = []

    for request, response in zip(availability_requests, availability_responses):
        origin_destinations = request.get("originDestinations", [])
        requested_ids = {od["id"] for od in origin_destinations}
        response_entries = response or []
        returned_ids = {entry.get("originDestinationId") for entry in response_entries}
        missing_ids = requested_ids - returned_ids

        for missing_id in sorted(missing_ids, key=int):
            od = next(od for od in origin_destinations if od["id"] == missing_id)
            unavailable_flights.append(
                {
                    "origin": od["originLocationCode"],
                    "destination": od["destinationLocationCode"],
                    "date": od["departureDateTime"]["date"],
                }
            )
    return unavailable_flights


def prune_unavailable_dates(
    travel_windows: list[dict[str, Any]],
    unavailable_flights: Iterable[dict[str, str]],
) -> list[dict[str, Any]]:
    unavailable_set = {
        (flight["origin"], flight["destination"], flight["date"]) for flight in unavailable_flights
    }

    for window in travel_windows:
        dates = window.get("dates", [])
        window["dates"] = [
            date
            for date in dates
            if (window["origin"], window["destination"], date) not in unavailable_set
        ]
    return travel_windows


def process_availability(
    *,
    amadeus: Client,
    travel_dates: list[dict[str, Any]],
    batch_size: int = 6,
) -> tuple[list[dict[str, Any]], list[Any], list[dict[str, str]]]:
    requests_path = data_dir_today / AVAILABILITY_REQUESTS_FILENAME
    responses_path = data_dir_today / AVAILABILITY_RESPONSES_FILENAME
    unavailable_path = data_dir_today / UNAVAILABLE_FLIGHTS_FILENAME

    if fetch_from_api:
        if refresh_data or not responses_path.exists():
            availability_requests = build_availability_requests(
                travel_dates,
                batch_size=batch_size,
            )
            _dump_json(requests_path, availability_requests)
            availability_responses = fetch_availability_batches(amadeus, availability_requests)
            _dump_json(responses_path, availability_responses)

            unavailable_flights = compute_unavailable_flights(
                availability_requests, availability_responses
            )
            _dump_json(unavailable_path, unavailable_flights)

    unavailable_flights = _load_json(unavailable_path) if unavailable_path.exists() else []
    return unavailable_flights


def iter_dates(start_date: str, end_date: str) -> Iterable[str]:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    delta = end - start
    for i in range(delta.days + 1):
        yield (start + timedelta(days=i)).strftime("%Y-%m-%d")


def get_travel_windows_with_dates(
    travel_windows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    windows = travel_windows or TRAVEL_WINDOWS
    schedule: list[dict[str, Any]] = []
    for window in windows:
        schedule.append(
            {
                "origin": window["origin"],
                "destination": window["destination"],
                "start_date": window["start_date"],
                "end_date": window["end_date"],
                "dates": list(iter_dates(window["start_date"], window["end_date"])),
            }
        )
    return schedule


def main() -> None:
    load_dotenv()

    travel_dates = get_travel_windows_with_dates()
    amadeus = build_amadeus_client(ENVIRONMENT)

    unavailable_flights = process_availability(
        amadeus=amadeus,
        travel_dates=travel_dates,
        batch_size=6,
    )

    if unavailable_flights:
        travel_dates = prune_unavailable_dates(travel_dates, unavailable_flights)

    _dump_json(data_dir_today / TRAVEL_DATES_FILENAME, travel_dates)

    print("Done availability data.")


if __name__ == "__main__":
    main()
