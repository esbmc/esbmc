# A pure function with a loop, called with a constant argument, is folded at
# conversion time by the consteval interpreter. Folding lets the assertion pass
# with --unwind 3, far below the 100 iterations the loop would otherwise need,
# proving the loop never reached the symbolic engine.
def total(n):
    s = 0
    i = 0
    while i < n:
        s += i
        i += 1
    return s


assert total(100) == 4950
