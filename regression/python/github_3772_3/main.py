# Annotation says str but function returns int - Python allows this
def get_num() -> int:
    return 10


def test() -> None:
    s: str = get_num()
    assert s == 10


test()
