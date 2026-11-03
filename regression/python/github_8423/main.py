def main() -> None:
    n = 1
    caught = False
    try:
        s = ",".join(["a", n])
    except TypeError:
        caught = True
    assert caught

    flag = True
    caught = False
    try:
        t = "".join((flag, "b"))
    except TypeError:
        caught = True
    assert caught

    a = "x"
    assert ",".join([a, "b"]) == "x,b"


main()
