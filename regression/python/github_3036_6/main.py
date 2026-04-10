def validate_hex_color(color_code: str) -> None:
    valid_chars = "0123456789abcdefABCDEF"
    # Generator checks if every character c exists in the valid_chars string
    assert all(c in valid_chars for c in color_code), "Error: Invalid hex character found."

def main() -> None:
    color = "1A2b3C"
    validate_hex_color(color)

main()
