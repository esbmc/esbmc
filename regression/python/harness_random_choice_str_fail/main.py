# The failing half of harness_random_choice_str. The first assertion is the
# true one: on an unfixed binary the result is mistyped and *it* is the claim
# that fails, so pinning the second violation is what makes this test bite.
import random


def main() -> None:
    c = random.choice("abc")
    assert c == "a" or c == "b" or c == "c"
    assert c == "z"


main()
