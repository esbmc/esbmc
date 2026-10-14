# An empty tuple has no member to select, so the inline fold reports it rather
# than emitting a claim. CPython raises IndexError here.
import random


def main() -> None:
    v = random.choice(())
    assert v is not None


main()
