# Exercises --python-irep2-adjust-only on an inline Optional (None | int). The
# Optional is a struct { is_none, anon_pad#, value }; the converter-built literal
# carries the value operands but not the padding operand, so under the sole
# adjuster python_adjust must insert it (S2 padding completion, now fired on an
# already-resolved struct type too). Without it the post-adjust invariant flags
# "2 operand(s) against 3 component(s)".
def maybe(flag: bool) -> int | None:
    return None if flag else 42


assert maybe(False) == 42
