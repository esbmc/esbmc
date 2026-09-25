# sorted() applies its key once per element, as CPython does; the insertion
# sort re-applied it in the shift loop, which an impure key could observe.
cnt = 0


def f(c: int) -> int:
    global cnt
    cnt += 1
    return c


l = [3, 1, 2]
l.append(0)
s = sorted(l, key=f)
assert cnt == 4
t = sorted(l, key=lambda c: f(c))
assert cnt == 8 and t[0] == 0 and t[3] == 3
