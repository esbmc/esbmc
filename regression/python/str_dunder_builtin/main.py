def main() -> None:
    # x.__str__() on a built-in scalar (int/float/bool/str) was rejected with
    # "Unsupported '__str__'". It now routes to str(x) — the same path the str()
    # builtin uses — while a user object's own __str__ is still used.
    x = 42
    assert x.__str__() == "42"
    f = 3.5
    assert f.__str__() == "3.5"
    b = True
    assert b.__str__() == "True"
    s = "hi"
    assert s.__str__() == "hi"
    assert (5).__str__() == "5"

    # "".join(str(x) for x in xs) lowers str(x) to x.__str__(); ints now work.
    assert "".join(str(n) for n in [1, 2, 3]) == "123"


main()
