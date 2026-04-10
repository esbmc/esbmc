def fib(n: int) -> int:
    if n <= 1:
        return n
    return fib(n - 1) - fib(n - 2)


def main() -> None:
    assert fib(5) == 5


main()
