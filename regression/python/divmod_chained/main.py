total = 12345

h, rem1 = divmod(total, 3600)
m, s = divmod(rem1, 60)

assert h == 3
assert m == 25
assert s == 45
assert total == h * 3600 + m * 60 + s
