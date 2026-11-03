# Calls through function values (#8351, #8355, #8404, #8475): a variable
# bound in either branch, a function an imported function returns, a list
# element, a parameter, an alias, a conditional expression and a function
# passed as an argument.
from mod import pick

calls: int = 0
n = nondet_int()


def b_f1() -> int:
    return 1


def b_f2() -> int:
    return 2


def l_f1() -> int:
    return 1


def o_f1(x: int) -> int:
    global calls
    calls = calls + 1
    return x


def o_f2(x: int) -> int:
    global calls
    calls = calls + 10
    return x + 1


def apply(f, a: int) -> int:
    r = f(a)
    return r


def inc(a: int) -> int:
    return a + 1


def dbl(a: int) -> int:
    return a * 2


def get_n():
    return n


def func(a):
    return a()


def ident(x: int) -> int:
    return x


def c_f() -> int:
    return 1


def c_g() -> int:
    return 2


def d_f1():
    return 42


def d_f2():
    return "b"


def one():
    return 1


def two():
    return 2


def choose(flag):
    if flag:
        return one
    return two


def k_f():
    return 1


def k_g():
    return 2


def call(a):
    return a()


if nondet_bool():
    g1 = b_f1
else:
    g1 = b_f2
r1 = g1()

g2 = pick(nondet_bool())
r2 = g2()

fs = [l_f1]
g3 = fs[0]
r3 = g3()

if nondet_bool():
    g4 = o_f1
else:
    g4 = o_f2
r4 = g4(5)

a1 = apply(inc, 1)

b5 = func(get_n)

g6 = ident
c6 = g6(n) if nondet_bool() else g6(1)

c7 = nondet_bool()
h7 = c_f if c7 else c_g

dfs = [d_f1, d_f2]

flag = nondet_bool()
b8 = choose(flag)()

c9 = call(k_f if nondet_bool() else k_g)

# A failing assert ends its path, so each check runs on a path of its own.
k = nondet_int()
if k == 0:
    assert r1 == 1
elif k == 1:
    assert r2 == 1
elif k == 2:
    assert r3 == 2
elif k == 3:
    assert calls == 1
elif k == 4:
    assert apply(dbl, 3) == 7
elif k == 5:
    assert isinstance(b5, str)
elif k == 6:
    assert c6 == n
elif k == 7:
    assert h7() == 1
elif k == 8:
    assert b8 == 1
elif k == 9:
    assert c9 == 1
elif type(b5) is int:
    assert False
