# github #9909: the type probe of `x = rows[2]` read the nested list unguarded.
def f() -> list[int]:
    rows = [[1]]
    try:
        x = rows[2]
    except IndexError:
        x = [0]
    return x


assert f() == [0]
