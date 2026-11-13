# A nested def returning a str or abs() of its enclosing method's parameter
class C:
    def f(self, flag, n):
        def nested():
            if flag:
                return "Hello"
            return abs(n)

        return nested()


x = nondet_int()
flag = nondet_bool()
b = C().f(flag, x)
if flag:
    assert b == "Hello"
elif x >= 0:
    assert b == x
