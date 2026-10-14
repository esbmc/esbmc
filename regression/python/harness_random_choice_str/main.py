# Model: src/python-frontend/models/random.py (choice_chars).
# REQUIRES: nothing.
# ENSURES: E1 - random.choice(s) over a str returns one of its characters.
import random


def main() -> None:
    c = random.choice("abc")
    assert c == "a" or c == "b" or c == "c"  # E1


main()
