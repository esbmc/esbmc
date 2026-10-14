# Issue #7769: IndexError is raised from two statements, and each escapes on
# some path. Each must be reported at its own line, not at line 0.
def main() -> None:
    a = [1, 2]
    i: int = nondet_int()
    if i > 100:
        a[i + 2] = 1
    else:
        a[i + 3] = 2


main()
