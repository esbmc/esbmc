class C:
    def __init__(self):
        self.d: dict[str, int] = {"a": 1}

c = C()

for k, v in c.d.items():
    assert v == 1
