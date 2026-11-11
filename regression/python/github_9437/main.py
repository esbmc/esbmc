# #9437: a negative non-literal index into a str counts from the end.
s: str = "abcd"
i: int = -1
assert s[i] == "d"
k: int = nondet_int()
__ESBMC_assume(k >= 1 and k <= 4)
c: str = s[-k]
assert c != "x"
try:
    j: int = -5
    e: str = s[j]
    raised = False
except IndexError:
    raised = True
assert raised
