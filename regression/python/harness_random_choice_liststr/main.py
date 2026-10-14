# Model: src/python-frontend/models/random.py (choice_str).
# ENSURES: E1 - choice over a list of strings returns one of its elements.
import random


def main() -> None:
    c = random.choice(["ab", "cd"])
    assert c == "ab" or c == "cd"  # E1


main()
