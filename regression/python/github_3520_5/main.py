def f():
    x = int
    if isinstance(x, str):
        assert False  # Should not reach here
    else:
        assert True  # Should reach here


f()
