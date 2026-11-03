def classify(n: int) -> int:
    match n:
        case 0:
            return 0
        case x if x < 0:
            return -x
        case 7 as seven:
            return seven * 2
        case other:
            return other + 100


k = nondet_int()
__ESBMC_assume(k > -1000 and k < 1000)
assert classify(k) != 14
