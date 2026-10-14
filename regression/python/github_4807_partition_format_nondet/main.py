# Regression for #4807: str.partition() / str.format() / str.format_map()
# on a non-constant receiver (or argument) used to abort GOTO conversion
# with "X() requires constant strings". The handlers now fall back to a
# sound symbolic value:
#   - partition()  -> ("", "", "") tuple of three empty strings
#   - format()     -> nondet char *
#   - format_map() -> nondet char *
# so function bodies emit cleanly.


def f_partition(s: str, sep: str):
    return s.partition(sep)


def f_format(fmt: str, x: int) -> str:
    return fmt.format(x)


def f_format_map(fmt: str, d):
    return fmt.format_map(d)


if __name__ == "__main__":
    # No call sites: the function bodies above are emitted into GOTO and
    # previously aborted conversion. Successful conversion + zero VCCs ==
    # verification succeeds.
    pass
