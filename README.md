# Business Class Seatmap Generator w/ Amadeus API

CLI utility for browsing Amadeus seat-map availability across a set of long-haul trips. It can pull live data directly from the Amadeus Self-Service APIs or replay previously captured fixtures to make it easy to study cabin layouts and track window-seat availability over time.

## Example output

![Example output: weekly grid of ASCII seat maps and window-seat ledger](docs/seatmaps.png)

_Figure: Example terminal output showing the weekly ASCII seat maps (Mon–Sun) and the date-sorted ledger of available window seats._

## Highlights
- Builds ASCII seat maps with wide-character awareness so layouts stay aligned even when using emoji markers.
- Supports multiple execution modes: live API calls, JSON fixtures, or pre-built pickle snapshots.
- Prints a week-by-week grid plus per-day window-seat summaries with optional pricing pulled from flight offers.

## Requirements
- Python 3.11+ (tested locally with 3.11)
- `pip install amadeus python-dotenv`
- Amadeus Self-Service API credentials (test and/or production)

## Local setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install amadeus python-dotenv
cp .env.template .env
```

Fill in the `.env` file with the credentials from your Amadeus developer account:

```
TEST_AMADEUS_CLIENT_ID=...
TEST_AMADEUS_CLIENT_SECRET=...
AMADEUS_CLIENT_ID=...
AMADEUS_CLIENT_SECRET=...
```

The script reads these via `python-dotenv`, so the `.env` file only needs to exist in the repo root.

## Execution modes
Set the `environment` constant near the top of `flight_search.py` to pick a data source:

| Value | Description |
| --- | --- |
| `e2e-pickle` (default) | Loads the serialized objects in `test/seatmaps.pkl`. Useful for quick demos and keeping historical captures. |
| `e2e` | Reads the handcrafted JSON fixtures in `test/seatmap.json` and enriches them with prices from `test/flight-offer.json`. |
| `test` | Calls the Amadeus *test* environment using `TEST_AMADEUS_CLIENT_*` credentials. |
| `production` | Calls the Amadeus *production* environment using `AMADEUS_CLIENT_*` credentials. |

When running against `test` or `production`, the script iterates over the configured `travel_windows` and caches every retrieved seat map into `test/seatmaps.pkl` for future offline use.

## Usage
```bash
python flight_search.py
```

Typical output is:
- A blank spacer, followed by a weekly grid (Monday–Sunday) of ASCII seat maps. Missing days show a `NO DATA` placeholder.
- A date-sorted ledger of available window seats that includes rounded prices when available.

Adjust `travel_windows` in `flight_search.py` if you need different routes or date ranges. The helper `iter_dates` already walks every day between the `start_date` and `end_date`.

## Fixtures
`test/` contains reproducible inputs:
- `flight-offer.json` – sample response from `flight_offers_search`.
- `seatmap.json` – sample payload returned by `amadeus.shopping.seatmaps.post`.
- `seatmaps.pkl` – pickled list of `SeatMap` instances captured from earlier runs.

Feel free to swap these with real captures when building demos or regression tests.

## Contributing
No special tooling is required beyond standard linting/formatting for Python scripts. Please keep fixtures anonymized and avoid committing sensitive traveler data.

## License
This project is released under the MIT License. See [LICENSE](LICENSE) for the full text.
