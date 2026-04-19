def foo() -> str:
    return bar()

def bar() -> str:
    return "bar"

assert foo() == "bar"
