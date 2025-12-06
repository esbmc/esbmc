def foo(x):
    def bar(x):
        if x == 1:
            return 1
        else:
            return bar(x - 1)
    return bar(x)

assert foo(3) == 1