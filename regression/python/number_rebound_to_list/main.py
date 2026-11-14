def scaled(n: int) -> int:
    m = n * 2
    n = [m, n]
    return n[0] + n[1]


def local() -> int:
    n = 5
    m = n * 2
    n = [m]
    return n[0] + m


def first_digits(arr: list[int]) -> int:

    def first_digit(n):
        if n < 0:
            n = -1 * n
        n = [int(i) for i in str(n)]
        return n[0]

    return sum([first_digit(i) for i in arr])


assert scaled(5) == 15
assert local() == 20
assert first_digits([]) == 0
assert first_digits([-42]) == 4
