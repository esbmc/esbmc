def foo(k):
    counts = [0] * k
    assert counts[k - 1] == 0


foo(2)
