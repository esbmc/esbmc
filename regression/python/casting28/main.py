assert ord('<') == 60
assert ord('5') == 53
assert ord('A') == 65
assert ord('a') == 97
assert ord('\n') == 10
assert ord('€') == 8364  # Euro sign, UTF-8 multibyte
assert ord('ÿ') == 255  # Latin-1 character
assert ord('\u20AC') == 8364  # Euro sign using Unicode escape
