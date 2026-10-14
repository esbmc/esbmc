# An instance method takes a receiver, a class instance. The entry harness gives
# every parameter an arbitrary value, and the scalar scope cannot invent an
# object no constructor built, so the target is refused rather than harnessed
# over a nondet pointer (#6938 P4.5). The @staticmethod decorator decides this,
# not the receiver's name (method_entry_nonself_receiver_fail).
#
# Its caller-dependent half works (method_contract).
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
