# The failing half of harness_random_choice_liststr; see the float twin.
import random


def main() -> None:
    c = random.choice(["ab", "cd"])
    assert c == "ab" or c == "cd"
    assert c == "zz"


main()
