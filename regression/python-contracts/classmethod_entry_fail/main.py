# `klass` is the class object rather than an instance, and the entry harness can
# invent neither, so a classmethod is refused for the same reason. Spelling the
# receiver anything but `cls` is what makes the decorator, not the name, decide.
class Counter:
    @classmethod
    def add(klass, k: int) -> int:
        __ESBMC_requires(k > 0 and k < 1000)
        __ESBMC_ensures(__ESBMC_return_value > 0)
        return k

def main() -> None:
    v: int = Counter.add(3)
    assert v == 3

main()
