# The same program with KeyError checked fails at the lookup.
def main() -> None:
    d = {"a": 1}
    k = "a" if nondet_bool() else "b"
    v = d[k]
    if k == "b":
        y = 10 // 0


main()
