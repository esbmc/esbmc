# Issue #5114: ESBMC used to model str.partition() as a 1-element result, which
# made this false assertion provable (unsound). The real length is 3, so the
# expected verdict is VERIFICATION FAILED.
def main() -> None:
    assert len("a.b.c".partition(".")) == 1


main()
