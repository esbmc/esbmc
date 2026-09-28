# The dispatch keys on the call's receiver, so an aliased import must still
# reach the str variant rather than the int-list model.
import random as rnd


def main() -> None:
    c = rnd.choice("abc")
    assert c == "a" or c == "b" or c == "c"


main()
