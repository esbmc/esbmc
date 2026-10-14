# The failing half of harness_random_sample_str; see the choice twin.
import random


def main() -> None:
    a = random.sample("abc", 2)
    assert a[0] == "a" or a[0] == "b" or a[0] == "c"
    assert a[0] == "z"


main()
