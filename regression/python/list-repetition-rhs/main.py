def foo():
    digits = [1, 2]
    zeros = (len(digits) - 1) * [0]
    assert zeros[0] == 0


foo()
