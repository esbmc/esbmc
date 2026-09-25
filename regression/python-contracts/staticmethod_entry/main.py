# A method has no `self` when it is static, so its whole signature is scalar and
# it is inside the v1 contract scope. --function used to search only the module
# body, so it could not name a method at all and reported the target missing.
class Counter:
    @staticmethod
    def add(k: int) -> int:
        __ESBMC_requires(k > 0 and k < 1000)
        __ESBMC_ensures(__ESBMC_return_value > 0)
        return k

def main() -> None:
    v: int = Counter.add(3)
    assert v == 3

main()
