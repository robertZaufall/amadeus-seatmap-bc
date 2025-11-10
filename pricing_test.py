import json
import os
from pathlib import Path

from amadeus import Client, ResponseError
from dotenv import load_dotenv

load_dotenv()

environment = "production"

sw_get_offers = False

fixtures_dir = Path(__file__).parent / "test"
pricing_request_path_2 = fixtures_dir / "pricing_request_2.json"
pricing_response_data_path_2 = fixtures_dir / "pricing_response_2.json"

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
        hostname=environment
    )

if sw_get_offers:
    try:
        search_response = amadeus.shopping.flight_offers_search.get(
            originLocationCode='MUC',
            destinationLocationCode='BKK',
            departureDate='2025-12-15',
            returnDate='2026-01-20',
            adults=1,
            travelClass='BUSINESS',
            includedAirlineCodes='TG',
            nonStop='true',
            max=1
        )
    except ResponseError as error:
        raise error        

    if not search_response.data:
        print("No flight offers found")
        exit()

    flight = search_response.data[0]
else:
    flight = json.load(open(fixtures_dir / "flight-offer_return.json"))


flight2 = json.loads(json.dumps(flight))
flight2['id'] = '2'

pricing_request = {
    'data': {
        'type': 'flight-offers-pricing',
        'flightOffers': [flight, flight2]
    }
}
json.dump(pricing_request, open(pricing_request_path_2, "w"), indent=2)

try:
    pricing_response = amadeus.post(
        '/v1/shopping/flight-offers/pricing',
        pricing_request
    )
    pricing_data = pricing_response.data
except ResponseError as error:
    print(f"Pricing Error: {error}")
    print(f"Error details: {error.response.body if hasattr(error, 'response') else 'No details'}")
    raise error

print("\nPricing result:")
print(json.dumps(pricing_data, indent=2))
json.dump(pricing_data, open(pricing_response_data_path_2, "w"), indent=2)

print("Done.")
