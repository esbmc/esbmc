# sample() has no tuple variant -- a model parameter cannot name one arity and
# one member type. The sequence type is reported instead of running the
# int-list model, which used to raise a spurious memory-safety claim (#7673).
import random


def main() -> None:
    r = random.sample((10, 20, 30), 2)
    assert len(r) == 2


main()
