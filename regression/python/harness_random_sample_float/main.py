# Model: src/python-frontend/models/random.py (sample_float).
# ENSURES: E1 - sample over a list of floats keeps the element type, so every
#          sampled value is one of the population's.
import random


def main() -> None:
    r = random.sample([1.5, 2.5, 3.5], 2)
    assert len(r) == 2
    assert r[0] == 1.5 or r[0] == 2.5 or r[0] == 3.5  # E1


main()
