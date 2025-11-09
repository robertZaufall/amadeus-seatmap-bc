from typing import Optional

ANSI_RESET = "\033[0m"

# Semantic token -> ANSI escape sequence mapping. Keep tokens descriptive so
# config.py can use readable names instead of raw escape sequences.
TOKEN_MAP: dict[str, str] = {
    # foreground
    "fg_black": "\033[30m",
    "fg_red": "\033[31m",
    "fg_green": "\033[32m",
    "fg_yellow": "\033[33m",
    "fg_grey": "\033[90m",
    "fg_white": "\033[97m",
    "fg_bright_black": "\033[90m",
    # background
    "bg_white": "\033[47m",
    "bg_bright_white": "\033[107m",
    "bg_green": "\033[42m",
    "bg_red": "\033[41m",
    "bg_grey": "\033[100m",
    "bg_yellow": "\033[43m",
    "bg_black": "\033[40m",
    "bg_light_blue": "\033[104m",
    "bg_dark_blue": "\033[44m",
    "bg_dark_red": "\033[101m",
    "bg_dark_green": "\033[102m",
}


def resolve(color_or_token: Optional[str]) -> Optional[str]:
    if not color_or_token:
        return None
    if color_or_token.startswith("\033["):
        return color_or_token
    return TOKEN_MAP.get(color_or_token, None)


def apply(color: Optional[str], text: str) -> str:
    resolved = resolve(color)
    if not resolved:
        return text
    return f"{resolved}{text}{ANSI_RESET}"


def strip_ansi(text: str) -> str:
    return text.replace(ANSI_RESET, "")
