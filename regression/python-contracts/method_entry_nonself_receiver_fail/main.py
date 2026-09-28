# Python does not fix the receiver's name: `this` here is as much an instance as
# `self` is in method_entry_instance_fail. Keying the refusal on the name would
# harness this one over a nondet receiver and report a proof for it.
class Counter:
    def add(this, k: int) -> int:
        __ESBMC_requires(k > 0 and k < 1000)
        __ESBMC_ensures(__ESBMC_return_value > 0)
        return k

def main() -> None:
    c = Counter()
    v: int = c.add(3)
    assert v == 3

main()
