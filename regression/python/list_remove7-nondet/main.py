x: int = nondet_int()

l = [1, 2, 3, 2, 4]

# Only allow valid removals to avoid ValueError
if x == 1 or x == 2 or x == 3 or x == 4:
    old_len = len(l)
    l.remove(x)

    # Length decreases exactly by 1
    assert len(l) == old_len - 1

    # List still contains valid elements
    i: int = 0
    while i < len(l):
        assert l[i] == 1 or l[i] == 2 or l[i] == 3 or l[i] == 4
        i += 1
