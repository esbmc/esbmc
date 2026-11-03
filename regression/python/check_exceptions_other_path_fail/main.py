# The unchecked ZeroDivisionError ends only its own path. The IndexError on
# the other branch is still found.
def main() -> None:
    b = nondet_int()
    l = [1, 2]
    if b < 5:
        q = 10 // b
    else:
        x = l[b]


main()
