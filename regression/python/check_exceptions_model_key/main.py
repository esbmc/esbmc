# A missing key raises KeyError, so with KeyError unchecked the path ends
# at the lookup and the ZeroDivisionError reachable only after it is never
# raised.
def main() -> None:
    d = {"a": 1}
    k = "a" if nondet_bool() else "b"
    v = d[k]
    if k == "b":
        y = 10 // 0


main()
