def f1() -> int:
    return 1


def run() -> None:
    if nondet_bool():
        g = f1
    r = g()
    assert r == 1


run()
