def to_bin(x: int) -> str:
    return bin(x)


# Negative variant: bin(3) is "0b11", never "0b10". Used to pin the
# __python_int_to_bin operational model: a regression that returns a
# nondet/garbage string would let this wrong expectation pass.
assert to_bin(3) == "0b10"
