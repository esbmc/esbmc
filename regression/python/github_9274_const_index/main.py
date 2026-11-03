class C:
    def f(self) -> str:
        return "a"


x = [C().f(), C().f()]
assert x[0] == "a"
assert x[1] == "a"
