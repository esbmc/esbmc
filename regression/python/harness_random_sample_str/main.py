# Model: src/python-frontend/models/random.py (sample_chars, sample_str).
# ENSURES: E1 - sample over a str yields k one-character strings.
#          E2 - sample over a list of strings yields k of its elements.
import random


def main() -> None:
    a = random.sample("abc", 2)
    assert len(a) == 2
    assert a[0] == "a" or a[0] == "b" or a[0] == "c"  # E1
    b = random.sample(["ab", "cd", "ef"], 2)
    assert len(b) == 2  # E2


main()
