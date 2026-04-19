import random

def main():
    result: int = random.getrandbits(8)
    assert result > 255

    x: int = random.getrandbits(0)
    assert x == 1

    y: int = random.getrandbits(2)
    assert y >= 4

if __name__ == "__main__":
    main()

