class C:
    def f(self) -> str:
        return "a"


x = [C().f(), C().f()]
for i in range(2):
    assert x[i] == "a"
