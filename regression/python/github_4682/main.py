def f(d):
    v = d.get("a")
    if isinstance(v, int):
        return v
    return None


data = {"a": 10}
r = f(data)
