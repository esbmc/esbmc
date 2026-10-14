def g(x: int) -> int:
    return x + 1


def outer(a: int) -> tuple[int, int]:
    # Tuple literal in return position whose elements are non-constant
    # function calls. Previously this embedded a code_function_call into the
    # GOTO RETURN struct and crashed with a SIGSEGV (issue #5036).
    return (g(a), g(a))


def main() -> None:
    x, y = outer(5)
    assert x == 7
    assert y == 6


main()
