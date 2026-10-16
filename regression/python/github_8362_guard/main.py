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
r = classify(k)
if k == 0:
    assert r == 0
elif k < 0:
    assert r == -k
elif k == 7:
    assert r == 14
else:
    assert r == k + 100
