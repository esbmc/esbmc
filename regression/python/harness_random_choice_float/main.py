# Model: src/python-frontend/models/random.py (choice_float).
# ENSURES: E1 - choice over a list of floats returns one of its elements.
import random


def main() -> None:
    c = random.choice([1.5, 2.5])
    assert c == 1.5 or c == 2.5  # E1


main()
