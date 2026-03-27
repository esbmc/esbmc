def foo():
    def bar():
        return 42
    return bar()

assert foo() == 42
