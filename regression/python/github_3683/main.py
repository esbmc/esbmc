class A:

    def f(self, d: dict[str, dict[str, dict[str, str]]]) -> dict[str, bool]:
        r: dict[str, bool] = {}
        for k, v in d["a"].items():
            if v["x"] == "y":
                r[k] = True
        return r


res = A().f({"a": {"p": {"x": "y"}}})

assert res["p"] == True
