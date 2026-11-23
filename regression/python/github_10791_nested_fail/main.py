def f() -> None:
    d = {"N": {"D": {"V": [1]}}}
    d["N"]["D"]["V"].append(2)
    assert d["N"]["D"]["V"][1] == 3


f()
