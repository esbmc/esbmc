r: range = range(4)
it = r.__iter__()
total: int = 0
for x in it:
    total = total + x
assert total == 6  # 0+1+2+3 = 6
