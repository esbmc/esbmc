def pascal(n):
    rows = [[1]]
    for r in range(1, n):
        row = []
        for c in range(0, r):
            upleft = rows[r - 1][c - 1] if c > 0 else 0
            upright = rows[r - 1][c] if c < r else 0
            row.append(upleft + upright)
        rows.append(row)
    return rows


n = nondet_int()
if n == 3:
    rows = pascal(n)
    for r in range(n):
        assert len(rows[r]) == r + 1
