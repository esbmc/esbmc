# github #9909
def f() -> list[int]:
    rows = [[1]]
    try:
        x = rows[2]
    except IndexError:
        x = [0]
    return x


assert f() == [1]
