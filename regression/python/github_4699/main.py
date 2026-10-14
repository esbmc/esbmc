def g(x: int) -> int:
    if x > 0:
        return x + 1
    return 0


def main() -> None:
    a: int = 5
    # List literal whose elements are non-constant-foldable function calls.
    # Previously this leaked a code_function_call into the SSA and crashed
    # the SMT backend with a SIGSEGV (issue #4699).
    w: list[int] = [g(a), g(a)]
    assert w[0] == 6
    assert w[1] == 6


main()
