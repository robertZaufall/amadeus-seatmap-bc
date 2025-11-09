"""Central configuration for seatmap rendering and search behavior."""

# Valid values: "production", "test", "e2e", "e2e-pickle"
ENVIRONMENT = "e2e-pickle"
#ENVIRONMENT = "e2e"
#ENVIRONMENT = "test"
#ENVIRONMENT = "production"

# Toggle displaying the price below each rendered seatmap block.
SHOW_SEATMAP_PRICE = False

# Seatmap render style ("compact" or "normal").
SEATMAP_OUTPUT_STYLE = "compact"
#SEATMAP_OUTPUT_STYLE = "normal"

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

TRAVEL_WINDOWS = [
    {
        "origin": "MUC",
        "destination": "BKK",
        "start_date": "2025-11-24",
        "end_date": "2025-12-20",
    },
    {
        "origin": "BKK",
        "destination": "MUC",
        "start_date": "2026-01-15",
        "end_date": "2026-01-30",
    },
]

ANSI_RESET = "\033[0m"

BORDER_COLORS = {
    # semantic tokens; resolved at apply-time by colors.resolve/apply
    "default": "fg_bright_black",
    "best": "fg_green",
    "worst": "fg_red",
}

STATUS_SYMBOLS = {
    "AVAILABLE": "🟪",
    "OCCUPIED": "❌",
    "BLOCKED": "⬛",
}

WINDOW_AVAILABLE_SYMBOL = "🟩"

COMPACT_BACKGROUND_COLORS = {
    "AVAILABLE_WINDOW": "bg_dark_green",
    "AVAILABLE": "bg_dark_blue",
    "OCCUPIED": "bg_dark_red",
    "BLOCKED": "bg_grey",
    "UNKNOWN": "bg_white",
}

COMPACT_SYMBOLS = {
    "AVAILABLE_WINDOW": " ",
    "AVAILABLE": " ",
    "OCCUPIED": " ",
    "BLOCKED": " ",
    "UNKNOWN": " ",
}

COMPACT_SYMBOL_COLORS = {
    "AVAILABLE_WINDOW": "",
    "AVAILABLE": "",
    "OCCUPIED": "fg_black",
    "BLOCKED": "fg_black",
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
    "compact_seatmap_heading": "\nCompact seatmaps:\n",
    "availability_heading": "\nAvailable window seats by date:",
    "price_label": "Price",
    "no_window_seats": "No window seats",
    "roundtrip_window_title": "Round-trip price heatmap (window-seat prices)",
    "roundtrip_all_title": "Round-trip price heatmap (all prices, bold+italic = window-seat available)",
    "roundtrip_title_template": "Round-trip price heatmap ({outbound_route} + {return_route})",
}
