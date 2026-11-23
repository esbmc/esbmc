import re

assert re.sub(r"[0-9]+", "#", "a12b") == "a#b"
