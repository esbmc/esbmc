n = 3
seen = [False] * n
seen[1] = True
assert not seen[0] and seen[1] and not seen[2]
assert seen[1] == True and seen[0] == False
flags = [True, False]
flags.append(1 < 2)
assert len(flags) == 3 and flags[2]
copy = list(flags)
copy[0] = not copy[0]
assert flags[0] and not copy[0]
part = flags[1:]
assert not part[0] and part[1]
k = 0
for f in flags:
    if f:
        k += 1
assert k == 2
assert seen[1] + 1 == 2
assert sorted(flags)[0] == False and sorted(flags)[2] == True

first = seen[0]
assert isinstance(first, bool) and not isinstance(flags[1] + 0, bool)
assert str(seen[1]) == "True"
