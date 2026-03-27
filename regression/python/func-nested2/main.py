def foo(x):
    def bar():
        return x
    return bar()

assert foo(1) == 1