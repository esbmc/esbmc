import random


def main() -> None:
    # Negative: choice never returns a value outside the sequence.
    v = random.choice([1, 2, 3])
    assert v == 99
main()
