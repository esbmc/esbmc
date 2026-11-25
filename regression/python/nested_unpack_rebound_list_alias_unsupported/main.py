pairs = [(1, 2)]
if nondet_bool():
    pairs = [(2, 1)]
alias = pairs
alias[0] = 2
(a, b), = pairs
