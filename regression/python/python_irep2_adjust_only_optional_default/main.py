# Exercises --python-irep2-adjust-only on a defaulted Optional parameter. The
# default None materialises an Optional struct at the call site; its type must
# match the parameter's declared type. adjust_type had no code_type arm, so a
# function signature's embedded struct stayed unpadded while the value literal
# was padded, and symex rejected the call ("argument type mismatch: got struct,
# expected struct"). The code_type arm pads argument/return types too.
def use(x: int | None = None) -> int:
    if x is None:
        return 0
    return x


assert use() == 0
