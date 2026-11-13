# Regression for #7434: the trace steps inside the list operational model are
# reported at list.c's line numbers, which point past the end of this file
# unless the witness also names the file each step belongs to. A typed list
# (#10451) takes no witness step inside list.c, so this keeps the untyped one.
def main() -> None:
    v = nondet_int()
    xs: list[int] = []
    xs.append(v)
    assert xs[0] != 8


main()
