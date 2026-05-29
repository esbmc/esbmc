# A list initialised with an element from a *derived* binding (alias or
# expression) and then mutated by indexed assignment inside a for loop
# crashed the GOTO converter with a json operator[] assertion: get_typet
# assumed the aliased RHS was a Constant carrying a nested "value" (#4909).
Y = 1
X = Y
a = [X]
for i in range(1):
    a[i] = -1
assert a[0] == -1

# Expression (BinOp) RHS variant from the same report.
b0 = 1 + 0
b = [b0]
assert b[0] == 1
