# str.join() of a generator expression / comprehension must join the runtime
# elements, not fall back to a nondet string. Regression for issue #5123
# (''.join(xor(x, y) for x, y in zip(a, b))).
def combine(a: str, b: str, sep: str) -> str:
    def pick(i: str, j: str) -> str:
        if i == j:
            return 'S'
        else:
            return 'D'
    return sep.join(pick(x, y) for x, y in zip(a, b))


def main() -> None:
    assert combine('abc', 'axc', '') == 'SDS'
    assert combine('abc', 'axc', '-') == 'S-D-S'


main()
