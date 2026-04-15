class uint256(int):
    pass


MOD = 123


def foo(x: uint256):
    return x % MOD


assert foo(uint256(5)) == 5
