def run():
    w = {"a": 10, "b": 20}
    xs = [1, 2, 3]
    assert w.__len__() == 2
    assert xs.__len__() == 3
    assert w.__contains__("a")
    assert xs.__getitem__(1) == 2


run()
