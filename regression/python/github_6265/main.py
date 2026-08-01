# github.com/esbmc/esbmc/issues/6265
# Regression guard: genuine list + list concatenation must still verify.
def main() -> None:
    a = [1, 2]
    b = [3, 4]
    c = a + b
    assert len(c) == 4 and c[0] == 1 and c[3] == 4

    d = a + [i for i in range(3)]
    assert len(d) == 5


main()
