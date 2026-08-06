# Exercises the --python-irep2-adjust-only recursive #is_padding restore. An
# `int | None` attribute is a struct nested inside its class struct, and
# add_padding pads component types before the enclosing one (padding.cpp), so the
# already-padded Optional was padded a second time when its pad member arrived
# unflagged from the IREP2 round-trip. The oversized layout then put the sibling
# attribute past the object ("dereference failure: Object accessed with illegal
# offset").
class Box:
    def __init__(self) -> None:
        self.x: int | None = None
        self.flag: int = 7


def main() -> None:
    b = Box()
    _ = b.x
    assert b.flag == 7


main()
