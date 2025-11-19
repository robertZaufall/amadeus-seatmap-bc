from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from time import sleep
from typing import Any, Iterable, Sequence

from amadeus import Client, ResponseError
from dotenv import load_dotenv

from config import ENVIRONMENT

fetch_from_api = True
refresh_data = False # get data even data is present from today

data_dir = Path(__file__).parent / "data"
data_dir.mkdir(parents=True, exist_ok=True)
data_dir_today = data_dir / datetime.now().strftime("%Y%m%d")
data_dir_today.mkdir(parents=True, exist_ok=True)


TRAVEL_DATES_FILENAME = "travel_dates.json"
AVAILABLE_FLIGHTS_FILENAME = "availability_responses.json"
FLIGHT_OFFER_ONEWAY_TEMPLATE_FILENAME = "flight_offer_oneway_template.json"
PRICING_REQUESTS_ONEWAY_FILENAME = "pricing_requests_oneway.json"
PRICING_RESPONSES_ONEWAY_FILENAME = "pricing_responses_oneway.json"
PRICING_RESPONSES_ONEWAY_SIMPLE_FILENAME = "pricing_responses_simple_oneway.json"


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


def get_travel_dates_oneway() -> list[list[dict[str, str]]]:
    if not (data_dir_today / TRAVEL_DATES_FILENAME).exists():
        raise RuntimeError("Travel dates data not found. Please run get_availability.py first.")
    travel_dates = _load_json(data_dir_today / TRAVEL_DATES_FILENAME)
    if len(travel_dates) != 2:
        raise RuntimeError("Expected travel dates for two travel windows.")

    window1 = travel_dates[0]
    window2 = travel_dates[1]

    travel_dates_oneway = []
    for dep_date in window1['dates']:
        travel_dates_oneway.append(
            {
                "origin": window1['origin'],
                "destination": window1['destination'],
                "start_date": dep_date,
                "end_date": dep_date,
            })
    for ret_date in window2['dates']:
        travel_dates_oneway.append(
            {
                "origin": window2['origin'],
                "destination": window2['destination'],
                "start_date": ret_date,
                "end_date": ret_date,
            })
    return travel_dates_oneway


def get_availability_dates() -> list[list[dict[str, str]]]:
    if not (data_dir_today / AVAILABLE_FLIGHTS_FILENAME).exists():
        raise RuntimeError("Availability flights data not found. Please run get_availability.py first.")
    availability_flights = _load_json(data_dir_today / AVAILABLE_FLIGHTS_FILENAME)

    travel_dates_oneway = []
    for batch in availability_flights:
        for flight in batch:
            departure = flight['segments'][0]['departure']
            arrival = flight['segments'][0]['arrival']
            travel_dates_oneway.append(
                {
                    "origin": departure['iataCode'],
                    "destination": arrival['iataCode'],
                    "start_date": departure['at'],
                    "end_date": arrival['at'],
                    "number": flight['segments'][0]['number'],
                })
    return travel_dates_oneway


def process_pricing(
    amadeus: Client,
    batch_size: int = 6,
) -> None:
    requests_path = data_dir_today / PRICING_REQUESTS_ONEWAY_FILENAME
    responses_path = data_dir_today / PRICING_RESPONSES_ONEWAY_FILENAME
    simple_responses_path = data_dir_today / PRICING_RESPONSES_ONEWAY_SIMPLE_FILENAME

    if fetch_from_api:
        if refresh_data or not responses_path.exists():

            travel_dates_oneway = get_availability_dates()
            flight_offer_oneway_template_path = data_dir / FLIGHT_OFFER_ONEWAY_TEMPLATE_FILENAME
            flight_offer_oneway_template = _load_json(flight_offer_oneway_template_path)

            flight_offers_oneway = []
            i = 0
            for travel_date in travel_dates_oneway:
                flight_offer_oneway = deepcopy(flight_offer_oneway_template)

                flight_offer_oneway["id"] = str(i % 6 + 1)
                last_ticketing_date = (datetime.now() + timedelta(days=6)).strftime("%Y-%m-%d")

                flight_offer_oneway["lastTicketingDate"] = last_ticketing_date
                flight_offer_oneway["lastTicketingDateTime"] = last_ticketing_date

                # itinerary 0
                flight_offer_oneway['itineraries'][0]['segments'][0]['departure']['iataCode'] = travel_date['origin']
                flight_offer_oneway['itineraries'][0]['segments'][0]['departure']['at'] = travel_date['start_date']
                flight_offer_oneway['itineraries'][0]['segments'][0]['arrival']['iataCode'] = travel_date['destination']
                flight_offer_oneway['itineraries'][0]['segments'][0]['arrival']['at'] = travel_date['end_date']
                flight_offer_oneway['itineraries'][0]['segments'][0]['arrival']['at'] = travel_date['end_date']

                flight_offer_oneway['itineraries'][0]['segments'][0]['number'] = travel_date['number']
                flight_offers_oneway.append(flight_offer_oneway)
                i += 1

            all_pricing_data = []
            pricing_requests = []
            
            # Process flight offers in batches of 6
            for batch_idx in range(0, len(flight_offers_oneway), batch_size):
                batch = flight_offers_oneway[batch_idx:batch_idx + batch_size]
                
                pricing_request = {
                    'data': {
                        'type': 'flight-offers-pricing',
                        'flightOffers': batch
                    }
                }

                pricing_requests.append(pricing_request)
                
            _dump_json(requests_path, pricing_requests)

            i = 0
            for pricing_request in pricing_requests:
                i += 1
                sleep(0.5)  # To avoid hitting rate limits
                try:
                    pricing_response = amadeus.post(
                        '/v1/shopping/flight-offers/pricing',
                        pricing_request
                    )
                    pricing_data = pricing_response.data
                    all_pricing_data.extend(pricing_data['flightOffers'] if 'flightOffers' in pricing_data else [])
                    print(f"Batch {i} completed successfully")
                except ResponseError as error:
                    print(f"Error details: {error.response.body if hasattr(error, 'response') else 'No details'}")
                    raise error

            # Combine all results
            combined_pricing_data = {
                'type': 'flight-offers-pricing',
                'flightOffers': all_pricing_data
            }
            
            print("\nAll pricing batches completed.")
            print(f"Total offers priced: {len(all_pricing_data)}")

            _dump_json(responses_path, combined_pricing_data)

    pricing_data = _load_json(responses_path)

    pricing_data_simple = []
    for flight in pricing_data['flightOffers']:
        flight_simple = {
            'outbound_route': flight['itineraries'][0]['segments'][0]['departure']['iataCode'] + "->" + flight['itineraries'][0]['segments'][0]['arrival']['iataCode'],
            'outbound_date': flight['itineraries'][0]['segments'][0]['departure']['at'].split("T")[0],
            'price': flight['price']['total'],
            'currency': flight['price']['currency'],
            'captured_at': datetime.now().strftime("%Y-%m-%d"),
        }
        pricing_data_simple.append(flight_simple)
    _dump_json(simple_responses_path, pricing_data_simple)


def main() -> None:
    load_dotenv()

    amadeus = build_amadeus_client(ENVIRONMENT)
    process_pricing(amadeus)

    print("Done pricing data.")


if __name__ == "__main__":
    main()
