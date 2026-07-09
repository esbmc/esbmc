# A ZeroDivisionError that is NOT caught (the handler only catches KeyError)
# propagates as an uncaught exception: VERIFICATION FAILED. This is the
# counterpart to try_except_zerodivision and guards against the raise being
# swallowed unconditionally.
def main() -> None:
    try:
        _ = 10 // 0
    except KeyError:
        pass


main()
