# The control that shows the harness reaches a method's parameters rather than
# merely resolving its name: `rv == 3` is true of the call main makes and false
# of an arbitrary k, so it verifies without --function and must fail with it.
class Counter:
    @staticmethod
    def add(k: int) -> int:
        __ESBMC_requires(k > 0 and k < 1000)
        __ESBMC_ensures(__ESBMC_return_value == 3)
        return k

def main() -> None:
    v: int = Counter.add(3)
    assert v == 3

main()
