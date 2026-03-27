class A:
    def f(self, d: dict[str, dict[str, int]]) -> dict:
        r = {}
        if "prop" in d:
            for k, v in d["prop"].items():
                r[k] = True
        return r


a = A()
a.f({"prop": {"x": 1, "y": 2}})
