# int and None share an element-type class but not a type, and folding them
# together let ESBMC prove `c is not None` on a tuple that holds None. The
# members' types must agree exactly, so this is reported (#7673).
import random


def main() -> None:
    c = random.choice((1, None))
    assert c == 1 or c is None


main()
