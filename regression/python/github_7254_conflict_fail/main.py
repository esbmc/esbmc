def compare(a, b):
    assert not (a < b)


def direct(p, q):
    compare(p, q)


def indirect(s, t):
    compare(s, t)


def forward(u, v):
    indirect(u, v)


direct(5, 5)
forward(1.2, 1.9)
