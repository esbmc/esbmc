d = {2: ("ab ", "cd ")}
board, solution = d[2]
board = board.strip()
expected = solution.rstrip()
assert board == "ab"
assert expected == "cd"
assert solution == "cd "


def check(k: int) -> None:
    b, s = d[k]
    b = b.strip()
    assert s == "cd "
    assert b == "ab"


check(2)
