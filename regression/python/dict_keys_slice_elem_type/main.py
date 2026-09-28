def run():
    w = {(1, 2): 10, (3, 4): 20}
    vs = list(w.keys())[0:2]
    u, v = vs[1]
    assert u == 3
    assert v == 4


run()
