# Issue #7769: two raise sites of one exception type, neither reachable.
# The uncaught-exception property is partitioned per site; every partition
# must hold.
def main() -> None:
    a = [1, 2]
    i: int = nondet_int()
    if i > 100:
        a[0] = 1
    else:
        a[1] = 2


main()
