class A:

    def f(self, d):
        if "a" in d:
            return {"x": 1}
        return {}


r = A().f({"a": 0})
assert r["x"] == 1
