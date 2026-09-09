# The failing half of harness_random_choice_tuple; see the float twin.
import random


def main() -> None:
    a = random.choice((10, 20, 30))
    assert a == 10 or a == 20 or a == 30
    assert a == 99


main()
