class A:

    def m(self, x: int) -> int:
        return x


class B(A):

    def m(self, x: int) -> int:
        return 10 // x


x = nondet_int()
__ESBMC_assume(x > 0 and x < 5)

total = 0
for o in [A(), B()]:
    total += o.m(x)
assert total == x + 10 // x

b = B()
xs = [b, A()]
for o in xs:
    total += o.m(x)
assert total == 2 * (x + 10 // x)
