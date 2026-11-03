class Node:
    pass


a = Node()
b = Node()
x = nondet_int()
__ESBMC_assume(x != 3)
assert ((x, a) < (3, b)) == (x < 3)
