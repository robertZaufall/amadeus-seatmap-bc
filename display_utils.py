from __future__ import annotations

import re
import unicodedata
from datetime import date

from colors import apply as apply_color
from config import (
    ANSI_RESET as CONFIG_ANSI_RESET,
    HEATMAP_EMPHASIS_STYLES,
    HEATMAP_HEADER_COLOR,
)


ANSI_ESCAPE_RE = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
EMPHASIS_CODE_MAP = {
    'bold': '\033[1m',
    'italic': '\033[3m',
}
WEEKDAY_SHORT_NAMES = ('Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su')


def extract_row_and_column(seat_number: str):
    row = ''.join(ch for ch in seat_number if ch.isdigit())
    column = ''.join(ch for ch in seat_number if ch.isalpha())
    return row, column


def char_display_width(character: str) -> int:
    """Return the display width of a single character."""
    return 2 if unicodedata.east_asian_width(character) in {'F', 'W'} else 1


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    if not text:
        return ''
    return ANSI_ESCAPE_RE.sub('', text)


def display_width(text: str) -> int:
    """Return the printable width of text accounting for wide characters."""
    visible = strip_ansi(text or '')
    return sum(char_display_width(ch) for ch in visible)


def resolve_seatmap_style(value: str | None) -> str:
    """Normalize the configured seatmap style to either 'ascii' or 'compact'."""
    normalized = (value or '').strip().lower()
    return 'compact' if normalized == 'compact' else 'ascii'


def apply_emphasis_styles(text: str, *, enabled: bool) -> str:
    """Wrap text with configured ANSI emphasis codes when enabled."""
    if not enabled or not text:
        return text
    segments: list[str] = []
    if HEATMAP_EMPHASIS_STYLES.get('italic'):
        segments.append(EMPHASIS_CODE_MAP['italic'])
    if HEATMAP_EMPHASIS_STYLES.get('bold'):
        segments.append(EMPHASIS_CODE_MAP['bold'])
    if not segments:
        return text
    prefix = ''.join(segments)
    return f"{prefix}{text}{CONFIG_ANSI_RESET}"


def apply_italic_only(text: str) -> str:
    """Wrap text with ANSI italic without inheriting bold config."""
    if not text:
        return text
    italic_code = EMPHASIS_CODE_MAP.get('italic')
    if not italic_code:
        return text
    return f"{italic_code}{text}{CONFIG_ANSI_RESET}"


def apply_bold_italic(text: str) -> str:
    """Apply both bold and italic emphasis regardless of config."""
    if not text:
        return text
    bold_code = EMPHASIS_CODE_MAP.get('bold')
    italic_code = EMPHASIS_CODE_MAP.get('italic')
    segments = [segment for segment in (italic_code, bold_code) if segment]
    if not segments:
        return text
    prefix = ''.join(segments)
    return f"{prefix}{text}{CONFIG_ANSI_RESET}"


def pad_to_width(text: str, width: int) -> str:
    """Pad or trim text so that its display width equals the provided width."""
    if width <= 0:
        return ''
    current_width = 0
    trimmed_parts: list[str] = []
    text = text or ''
    idx = 0
    while idx < len(text):
        if text[idx] == '\x1b':
            match = ANSI_ESCAPE_RE.match(text, idx)
            if match:
                trimmed_parts.append(match.group(0))
                idx = match.end()
                continue
        ch = text[idx]
        ch_width = char_display_width(ch)
        if current_width + ch_width > width:
            break
        trimmed_parts.append(ch)
        current_width += ch_width
        idx += 1
    result = ''.join(trimmed_parts)
    if current_width < width:
        result += ' ' * (width - current_width)
    return result


def pad_to_width_centered(text: str, width: int) -> str:
    """Pad text to the provided width while keeping the content centered."""
    if width <= 0:
        return ''
    trimmed = pad_to_width(text, width)
    trimmed_content = trimmed.rstrip()
    content_width = display_width(trimmed_content)
    if content_width >= width:
        return trimmed_content
    padding = width - content_width
    left_padding = padding // 2
    right_padding = padding - left_padding
    return f"{' ' * left_padding}{trimmed_content}{' ' * right_padding}"


def weekday_short_name(value: date) -> str:
    """Return a two-letter weekday abbreviation starting on Monday."""
    return WEEKDAY_SHORT_NAMES[value.weekday()]


def apply_heatmap_header_color(text: str) -> str:
    """Color header cells using the configured darker grey tone."""
    if not text or not HEATMAP_HEADER_COLOR:
        return text
    return apply_color(HEATMAP_HEADER_COLOR, text)
