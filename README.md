# Business Class Seatmap Generator w/ Amadeus API

CLI utility for browsing Amadeus seat-map availability across a set of long-haul trips. It can pull live data directly from the Amadeus Self-Service APIs or replay previously captured fixtures to make it easy to study cabin layouts and track window-seat availability over time.

## Example output

![Example output: weekly grid of ASCII seat maps](docs/seatmaps_compact.png)
_Figure: Example terminal output showing the weekly ASCII seat maps (Mon–Sun)._

![Example output: window-seat ledger](docs/window_seats.png)
_Figure: Example terminal output showing the date-sorted ledger of available window seats including calendar heatmap for the price._

![Example output: price heatmaps](docs/price_heatmaps.png)
_Figure: Example terminal output showing the price heatmaps for all flight combinations - one for windows seats and one for all._

![Example output: weekly grid of ASCII seat maps - alternative layout](docs/seatmaps.png)
_Figure: Example terminal output showing the weekly ASCII seat maps (Mon–Sun) in an alternative layout (not current!)._

## Highlights
- Builds ASCII seat maps with wide-character awareness so layouts stay aligned even when using emoji markers.
- Highlights the lowest fare per route with a green border so deals stand out immediately.
- Supports multiple execution modes: live API calls, JSON fixtures, or pre-built pickle snapshots.
- Prints a week-by-week grid plus per-day window-seat summaries with optional pricing pulled from flight offers.
- Generates calendar heatmaps (per route and round-trip) to make fare trends obvious when scanning many dates.

## Requirements
- Python 3.11+ (tested locally with 3.11)
- `pip install -r requirements.txt`
- Amadeus Self-Service API credentials (test and/or production)

## Local setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
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

## Configuration
Most runtime knobs live in `config.py`, so you rarely need to edit `flight_search.py` directly. Notable settings:
- `ENVIRONMENT` picks the data source (`production`, `test`, `e2e`, `e2e-pickle`).
- `TRAVEL_WINDOWS` lists the routes/date ranges that should be fetched and rendered.
- `FLIGHT_SEARCH_FILTERS` stores the request arguments (`travel_class`, `non_stop`, `included_airline_codes`, etc.) passed to the Amadeus `flight_offers_search` endpoint.
- Visual toggles such as `SEATMAP_OUTPUT_STYLE`, `SHOW_SEATMAP_PRICE`, `HEATMAP_EMPHASIS_STYLES`, `STATUS_SYMBOLS`, and `WINDOW_AVAILABLE_SYMBOL` control how the ASCII/emoji output looks.
- Currency/price decoration (`CURRENCY_SYMBOLS`, `BORDER_COLORS`, etc.) reference the semantic tokens defined in `colors.py`; extend `colors.TOKEN_MAP` if you need custom ANSI sequences.

## Execution modes
Set `ENVIRONMENT` in `config.py` to pick a data source:

| Value | Description |
| --- | --- |
| `e2e-pickle` (default) | Loads the serialized objects in `test/seatmaps.pkl`. Useful for quick demos and keeping historical captures. |
| `e2e` | Reads the handcrafted JSON fixtures in `test/seatmap.json` and enriches them with prices from `test/flight-offer.json`. |
| `test` | Calls the Amadeus *test* environment using `TEST_AMADEUS_CLIENT_*` credentials. |
| `production` | Calls the Amadeus *production* environment using `AMADEUS_CLIENT_*` credentials. |

When running against `test` or `production`, the script iterates over the configured `TRAVEL_WINDOWS` and caches every retrieved seat map into `test/seatmaps.pkl` for future offline use.

## Travel windows & filters
Define the routes and date ranges you care about by editing the `TRAVEL_WINDOWS` list inside `config.py`:

```python
TRAVEL_WINDOWS = [
    {
        "origin": "MUC",
        "destination": "BKK",
        "start_date": "2025-11-24",
        "end_date": "2025-12-20",
    },
    # add more windows as needed
]
```

Each window is inclusive, so the script requests seat maps for every day in the range. The first two windows also drive the combined round-trip heatmaps (outbound vs. return).

Flight-offer search filters live in `config.FLIGHT_SEARCH_FILTERS`, so you can tweak cabin class, airline, or connection rules without touching the main script. The defaults target non-stop business-class flights on Thai Airways:

```python
FLIGHT_SEARCH_FILTERS = {
    "travel_class": "BUSINESS",
    "non_stop": "true",
    "included_airline_codes": "TG",
    # add "adults", "children", etc. if needed
}
```

Any key/value pairs in this dict get splatted into the `SeatMaps.fetch` call.

## Usage
```bash
python flight_search.py
```

Typical output is:
- A blank spacer, followed by a weekly grid (Monday–Sunday) of ASCII seat maps. Missing days show a `NO DATA` placeholder and the cheapest fare on each route is highlighted.
- A route-by-route window-seat ledger that includes rounded prices, per-date availability notes, and a small calendar heatmap to hint at relative fares.
- (Optional) Two round-trip matrices that add outbound + return fares together—one based on window-seat pricing and one using any available fare—whenever at least two travel windows are defined.

Adjust `TRAVEL_WINDOWS` in `config.py` if you need different routes or date ranges. The helper `iter_dates` already walks every day between the `start_date` and `end_date`.

## Output details
- **Weekly seat-map grid** – One block per day, grouped by week. Missing data shows a `NO DATA` placeholder, and the absolute lowest fare per route gets a green border plus a rounded price label at the bottom of the block.
- **Route availability boxes** – After the grid, every route receives a bordered box that lists the available window seats per date, prefixed with a relative fare symbol and paired with a mini calendar heatmap for additional visual context.
- **Round-trip price heatmaps** – If you provide at least two travel windows (outbound + inbound), the script prints two combined matrices: one based on window-seat fares only and another that considers any price returned by the offer search.

## Roundtrip price database
Each time the round-trip heatmaps are generated, the script also persists the combined price matrix to a local SQLite file (`roundtrip_prices.db`). Every record stores:
* outbound/return routes and dates
* summed fare (as text) and currency
* the capture timestamp that produced the price
* whether window seats were available on the outbound and return legs when that fare was recorded

When a new run encounters data already in the database, prices are replaced only if the fresh capture is newer, but the window-availability flags are always backfilled when they were missing previously (this keeps legacy pickle snapshots useful). You can inspect the DB with the built-in CLI:

```bash
sqlite3 roundtrip_prices.db 'SELECT * FROM combination_prices LIMIT 5;'
```

## Fixtures
`test/` contains reproducible inputs:
- `flight-offer.json` – sample response from `flight_offers_search`.
- `seatmap.json` – sample payload returned by `amadeus.shopping.seatmaps.post`.
- `seatmaps.pkl` – pickled list of `SeatMap` instances captured from earlier runs.

Feel free to swap these with real captures when building demos or regression tests.

## Refreshing cached seat maps
Whenever you run in `test` or `production` mode, newly fetched seat maps automatically overwrite `test/seatmaps.pkl`. This keeps the offline (`e2e-pickle`) mode in sync with your latest captures. Delete the file or run the script again to regenerate it whenever you want a fresh snapshot.

## Contributing
No special tooling is required beyond standard linting/formatting for Python scripts. Please keep fixtures anonymized and avoid committing sensitive traveler data.

## License
This project is released under the MIT License. See [LICENSE](LICENSE) for the full text.
