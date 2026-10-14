# `self` is an ordinary parameter name in a static method, so the decorator
# admits this target where the old name-based test refused it. The frontend
# still binds `self` and `cls` to the enclosing class wherever they appear, so
# report that limitation rather than tripping an assertion downstream.
class Counter:
    @staticmethod
    def add(self: int) -> int:
        __ESBMC_requires(self > 0 and self < 1000)
        __ESBMC_ensures(__ESBMC_return_value > 0)
        return self

def main() -> None:
    v: int = Counter.add(3)
    assert v == 3

main()
