def run():
    w = {"a": 10, "b": 20}
    # 20 is w["b"], not w["a"]: the call must really read the dict, not just
    # avoid raising AttributeError.
    assert w.__getitem__("a") == 20


run()
