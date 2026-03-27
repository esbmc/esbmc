import math

def test_logarithms():
    try:
        math.log(0)
        assert False, "Expected ValueError"
    except ValueError:
        pass

test_logarithms()
