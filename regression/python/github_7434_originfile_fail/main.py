# Regression for #7434: the trace steps inside the list operational model are
# reported at list.c's line numbers, which point past the end of this file
# unless the witness also names the file each step belongs to.
def main() -> None:
    xs: list[int] = []
    xs.append(7)
    assert xs[0] == 8


main()
