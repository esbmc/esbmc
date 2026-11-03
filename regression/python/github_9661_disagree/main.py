class A:
    def f(self, x):
        return x


def call_int(obj):
    return obj.f(1)


def call_str(obj):
    return obj.f("s")


assert call_int(A()) == 1
assert call_str(A()) == "s"
