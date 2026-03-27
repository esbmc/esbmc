import math

def test_logarithms():
    try:
        math.log(1)
        assert False, "Expected ValueError"
    except ValueError:
        pass

test_logarithms()
