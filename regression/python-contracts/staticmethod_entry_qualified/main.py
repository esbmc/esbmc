# The same target named as Class.method, which is the spelling needed when more
# than one class defines it.
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
