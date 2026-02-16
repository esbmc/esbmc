import random

def main():
    result: int = random.getrandbits(8)
    assert result >= 0 and result <= 255
    
    result2: int = random.getrandbits(4)
    assert result2 >= 0 and result2 <= 15
    
    result3: int = random.getrandbits(1)
    assert result3 >= 0 and result3 <= 1
    
    result4: int = random.getrandbits(16)
    assert result4 >= 0 and result4 <= 65535

    x: int = random.getrandbits(0)
    assert x == 0

    y: int = random.getrandbits(1)
    assert y == 0 or y == 1

    z: int = random.getrandbits(3)
    assert z >= 0
    assert z <= 7

    try:
        result5: int = random.getrandbits(-1)
        assert False  # should not reach here
    except ValueError:
        assert True
    
    try:
        result6: int = random.getrandbits(-10)
        assert False  # should not reach here
    except ValueError:
        assert True

if __name__ == "__main__":
    main()
