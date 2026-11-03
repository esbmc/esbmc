# Reading an element of a list parameter used to leave the branch on it
# symbolic, so the loop bounded by n never stopped unwinding.
def f(a: list[int]) -> int:
    n = 0
    if a[1] < 3:
        n = 2
    k = 0
    while k < n:
        k += 1
    return k


assert f([3, 1]) == 2
