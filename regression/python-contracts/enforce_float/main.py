# Caller-dependent tier: see enforce_float_entry for the caller-independent
# claim. x / 2.0 < x does not hold for every x > 0.0.
def half(x: float) -> float:
    __ESBMC_requires(x > 0.0)
    __ESBMC_ensures(__ESBMC_return_value < x)
    return x / 2.0

def main() -> None:
    z: float = half(4.0)
    assert z < 4.0

main()
