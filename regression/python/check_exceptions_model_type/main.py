# Dividing a str raises TypeError inside the scalar model, which checks it
# with an assert. With TypeError unchecked the path ends there, so the
# ZeroDivisionError reachable only after it is never raised.
def main() -> None:
    if nondet_bool():
        r = 1
    else:
        r = "s"
    h = r / 2
    if isinstance(r, str):
        y = 10 // 0


main()
