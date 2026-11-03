d = {2: ("ab ", "cd ")}
board, solution = d[2]
board = board.strip()
expected = solution.rstrip()
assert expected == "cd "
