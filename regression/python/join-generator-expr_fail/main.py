# Negative variant: a wrong expected join result must not verify (guards against
# the old nondet-string fallback silently "passing").
def combine(a: str, b: str) -> str:
    def pick(i: str, j: str) -> str:
        if i == j:
            return 'S'
        else:
            return 'D'
    return ''.join(pick(x, y) for x, y in zip(a, b))


def main() -> None:
    # combine('abc', 'axc') == 'SDS'; asserting otherwise must fail.
    assert combine('abc', 'axc') == 'SSS'


main()
