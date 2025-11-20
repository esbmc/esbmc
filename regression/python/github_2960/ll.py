from md import Foo, Bar

def create(s: str) -> Foo | Bar:
    if s == "Foo":
        return Foo(s)
    elif s == "Bar":
        return Bar(s)
    else:
        raise ValueError("Invalid class name")
