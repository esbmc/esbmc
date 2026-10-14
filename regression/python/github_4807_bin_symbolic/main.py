def to_bin(x: int) -> str:
    return bin(x)


# Symbolic-int bin(): the previous frontend path required an AST integer
# literal and threw TypeError for variable arguments. With the
# __python_int_to_bin operational model these calls now produce the
# correct string at symex time.
assert to_bin(0) == "0b0"
assert to_bin(1) == "0b1"
assert to_bin(3) == "0b11"
assert to_bin(15) == "0b1111"
assert to_bin(-2) == "-0b10"
