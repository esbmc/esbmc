# The failing half of harness_random_choice_float. The first assertion is the
# true one: on an unfixed binary the result is mistyped and *it* is the claim
# that fails, so pinning the second violation is what makes this test bite.
# The false value is outside the list so CPython rejects it deterministically.
import random


def main() -> None:
    c = random.choice([1.5, 2.5])
    assert c == 1.5 or c == 2.5
    assert c == 9.5


main()
