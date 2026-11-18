def f(flag: bool) -> None:
    d = {"Not": {"D": {"V": ["a"]}}}
    if flag:
        d = {"D": {"V": ["b"]}}
    if "Not" in d:
        d["Not"]["D"]["V"].append("c")


f(nondet_bool())
