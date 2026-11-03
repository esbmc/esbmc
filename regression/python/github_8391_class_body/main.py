def used() -> int:
    return 1


class A:
    t = [1, *[2]]


assert used() == 1
