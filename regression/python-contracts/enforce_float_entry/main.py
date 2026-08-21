# The float half of the entry-harness tier. Bounded away from the values where
# x / 2.0 < x stops holding, so the postcondition is a claim about every
# admitted x rather than about the argument main passes.
def half(x: float) -> float:
    __ESBMC_requires(x > 1.0 and x < 1000.0)
    __ESBMC_ensures(__ESBMC_return_value < x)
    return x / 2.0

def main() -> None:
    z: float = half(4.0)
    assert z < 4.0

main()
