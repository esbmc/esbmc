# A method contract can only be checked against the callers the program happens
# to contain: --function will not name a method, so there is no entry harness
# for one and the caller-independent claim cannot be made at all. `add`,
# `Counter.add` and `Counter@add` are all rejected with "Function not found".
#
# The caller-dependent half works (method_contract).
class Counter:
    def __init__(self) -> None:
        self.n: int = 0

    def add(self, k: int) -> int:
        __ESBMC_requires(k > 0 and k < 1000)
        __ESBMC_ensures(__ESBMC_return_value > 0)
        return k

def main() -> None:
    c = Counter()
    v: int = c.add(3)
    assert v == 3

main()
