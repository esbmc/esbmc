def run():
    d = {1: 30, 2: 10}
    d[3] = 20
    ps = sorted(d.items(), key=sum)
    assert ps[0][0] == 2


run()
