def main() -> None:
    x: int = nondet_int()
    y: int = nondet_int()

    __VERIFIER_assume(-10 <= x <= 10)
    __VERIFIER_assume(-10 <= y <= 10)

    z: int = 0
    w: int = 0

    if x > 0 and y > 0:
        z = x + y
        assert z >= x
        assert z >= y
    else:
        w = x * x + y * y
        assert w >= 0

main()
