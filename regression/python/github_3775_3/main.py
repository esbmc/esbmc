class Base:
    pass

class Mid(Base):
    pass

class Child(Mid):
    def label(self) -> str:
        return "hello"

def test():
    c = Child()
    y: int = c.label()
    assert y == "hello"

test()
