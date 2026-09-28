# After m.pop() empties the list, m.append(7) retypes it to int: reading the
# stale (popped) float type here must not make this assertion pass.
m = [4.5]
m.pop()
m.append(7)
assert m[0] == 4.5
