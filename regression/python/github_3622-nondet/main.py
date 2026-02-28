lst = []
n = nondet_int()
__ESBMC_assume(0 <= n <= 3)

i = 0
while i < n:
    lst.append(nondet_int())
    i += 1

if len(lst) > 0:
    v = lst[-1]  # or constrain v to equal lst[-1]
    assert lst[-1] == v  # trivially true
