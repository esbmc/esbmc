class Box:
    def __init__(self, v: int):
        self.v = v


def test() -> None:
    # Reading a list value from an unannotated dict literal must return a
    # list, not a string (the default cast before this fix treated the
    # value as char* and routed len() to strlen()).
    a = {0: [1, 2, 3]}
    assert len(a[0]) == 3
    assert a[0] == [1, 2, 3]

    # Nested dict read through an intermediate variable: `inner` must be
    # typed as dict so that subsequent subscripts route to dict handling
    # instead of generic list indexing.
    d = {"k": {"x": 7}}
    inner = d["k"]
    assert inner["x"] == 7

    # Direct attribute access on a dict subscript of a class-instance value.
    # Needs: (1) Attribute handling for Subscript values and (2) dict value
    # resolution that recognises user-defined class struct types.
    cb: dict[int, Box] = {}
    cb[0] = Box(7)
    assert cb[0].v == 7


test()
