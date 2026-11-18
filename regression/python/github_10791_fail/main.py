def f(flag: bool) -> None:
    d = {"n": {"v": ["a"]}}
    if flag:
        d = {"v": {"w": ["b"]}}
    if "n" in d:
        d["n"]["v"].append("c")
        assert flag


f(nondet_bool())
