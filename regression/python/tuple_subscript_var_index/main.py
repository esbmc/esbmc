# Exercises the non-constant tuple-index select-chain path in tuple_handler,
# where #5468 and #5472 each declared idx2t/idxnorm2 in the same scope (a
# redefinition that broke the build). With the duplicate removed, a tuple
# subscript by a runtime index resolves correctly.
def get(t: tuple[int, int, int], i: int) -> int:
    return t[i]


def main() -> None:
    t = (10, 20, 30)
    assert get(t, 1) == 20
    assert get(t, 2) == 30


main()
