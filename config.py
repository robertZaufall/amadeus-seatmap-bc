"""Central configuration for seatmap rendering and search behavior."""

from datetime import date, timedelta

# Valid values: "production", "test", "e2e", "e2e-pickle"
#ENVIRONMENT = "e2e-pickle"
#ENVIRONMENT = "e2e"
#ENVIRONMENT = "test"
ENVIRONMENT = "production"

def _build_travel_windows(today: date | None = None) -> list[dict[str, str]]:
    base = today or date.today()
    first_start = base + timedelta(days=3)
    first_end = first_start + timedelta(days=20)  # 3 weeks inclusive

    second_start = first_start + timedelta(weeks=6)
    second_end = second_start + timedelta(days=24)  # 3.5 weeks inclusive

    def _iso(d: date) -> str:
        return d.strftime("%Y-%m-%d")

    return [
        {
            "origin": "MUC",
            "destination": "BKK",
            "start_date": _iso(first_start),
            "end_date": _iso(first_end),
        },
        {
            "origin": "BKK",
            "destination": "MUC",
            "start_date": _iso(second_start),
            "end_date": _iso(second_end),
        },
    ]


TRAVEL_WINDOWS = _build_travel_windows()

# Default filters applied when requesting flight offers from Amadeus.
FLIGHT_SEARCH_FILTERS = {
    "travel_class": "BUSINESS",
    "non_stop": "true",
    "included_airline_codes": "TG",
}

# Toggle displaying the price below each rendered seatmap block.
SHOW_SEATMAP_PRICE = False

# Seatmap render style ("compact", "normal", or "both").
SEATMAP_OUTPUT_STYLE = "both"
#SEATMAP_OUTPUT_STYLE = "compact"
#SEATMAP_OUTPUT_STYLE = "normal"

# For compact view
SUPPRESS_COMPACT_SECOND_HEADER = True

# Controls whether highlighted heatmap cells render bold and/or italic.
HEATMAP_EMPHASIS_STYLES = {
    "bold": True,
    "italic": True,
}

CURRENCY_SYMBOLS = {
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "CNY": "¥",
    "THB": "฿",
    "AUD": "$",
    "CAD": "$",
    "SGD": "$",
}

ANSI_RESET = "\033[0m"

BORDER_COLORS = {
    # semantic tokens; resolved at apply-time by colors.resolve/apply
    "default": "fg_bright_black",
    "best": "fg_green",
    "worst": "fg_red",
}

# 🟦 / 🟪 / 🟥 / 🟧 / 🟨 / 🟩 / 🟫
STATUS_SYMBOLS = {
    "AVAILABLE": "🟦",
    "OCCUPIED": "❌",
    "BLOCKED": "⬛",
}

WINDOW_AVAILABLE_SYMBOL = "🟩"

COMPACT_BACKGROUND_COLORS = {
    "AVAILABLE_WINDOW": "bg_dark_green",
    "AVAILABLE": "bg_dark_blue",
    "OCCUPIED": "",
    "BLOCKED": "",
    "UNKNOWN": "bg_white",
}

COMPACT_SYMBOLS = {
    "AVAILABLE_WINDOW": " ",
    "AVAILABLE": " ",
    "OCCUPIED": "✘",
    "BLOCKED": "✘",
    "UNKNOWN": " ",
}

COMPACT_SYMBOL_COLORS = {
    "AVAILABLE_WINDOW": "",
    "AVAILABLE": "",
    "OCCUPIED": "fg_red",
    "BLOCKED": "fg_grey",
    "UNKNOWN": "",
}

HEATMAP_SYMBOLS = {
    "min": "🟩",
    "default": "🟨",
    "max": "🟥",
}

HEATMAP_COLORS = {
    "min": BORDER_COLORS["best"],
    "default": "fg_yellow",
    "max": BORDER_COLORS["worst"],
}

HEATMAP_HEADER_COLOR = "fg_grey"

STATIC_LABELS = {
    "compact_seatmap_heading": "\nSeatmaps:\n",
    "normal_seatmap_heading": "\nSeatmaps (normal view):\n",
    "availability_heading": "\nAvailable window seats by date:",
    "price_label": "Price",
    "no_window_seats": "No window seats",
    "roundtrip_window_title": "Round-trip price heatmap (window-seat prices)",
    "roundtrip_all_title": "Round-trip price heatmap (all prices, bold+italic = window-seat available)",
    "roundtrip_return_prices_title": "Round-trip price heatmap (return prices data)",
    "roundtrip_title_template": "Round-trip price heatmap ({outbound_route} + {return_route})",
}
