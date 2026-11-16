# #9437: a negative non-literal index into a str counts from the end.
s: str = "abcd"
k: int = nondet_int()
__ESBMC_assume(k >= 1 and k <= 4)
c: str = s[-k]
assert c != "a"
