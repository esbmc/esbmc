# The failing half of harness_random_sample_float; see the choice twin.
import random


def main() -> None:
    r = random.sample([1.5, 2.5, 3.5], 2)
    assert r[0] == 1.5 or r[0] == 2.5 or r[0] == 3.5
    assert r[0] == 9.5


main()
