# ascii() of a runtime non-ASCII str returns a string, not TypeError (#7559).
def f(s: str) -> None:
    try:
        ascii(s)
    except TypeError:
        assert False


f("\u00e9")
