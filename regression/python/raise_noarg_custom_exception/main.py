# Raising a no-argument custom exception whose class inherits __init__ (e.g.
# class E(Exception): pass; raise E()) previously crashed ESBMC — get_expr
# lowered E() to a bare cpp-throw side-effect, which was then wrapped in another
# cpp-throw and aborted value_set's make_member. It is now caught normally.
class E(Exception):
    pass


def main() -> None:
    caught = 0
    try:
        raise E()
    except E:
        caught = 1
    assert caught == 1

    try:
        raise E()
    except E as e:
        caught = 2
    assert caught == 2

    # Caught by a base class too.
    try:
        raise E()
    except Exception:
        caught = 3
    assert caught == 3


main()
