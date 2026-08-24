# A bare name two classes define says nothing about which was meant. Picking one
# would harness a different function than the one asked for, so the ambiguity is
# reported and the qualified spelling requested.
class A:
    @staticmethod
    def add(k: int) -> int:
        __ESBMC_requires(k > 0 and k < 1000)
        __ESBMC_ensures(__ESBMC_return_value > 0)
        return k

class B:
    @staticmethod
    def add(k: int) -> int:
        __ESBMC_requires(k > 0 and k < 1000)
        __ESBMC_ensures(__ESBMC_return_value > 0)
        return k

def main() -> None:
    v: int = A.add(3)
    assert v == 3

main()
