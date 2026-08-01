# Exercises int.to_bytes() under --python-irep2-adjust-only. The byte-extraction
# shift must be built as `lshr`, not the `shr` placeholder: only
# clang_c_adjust::adjust_expr_shifts resolves `shr`, and migrate_expr has no
# `shr` arm, so under the hop-off a surviving `shr` aborts with "migrate expr
# failed" before any verdict is reached.
def main() -> None:
    b = (258).to_bytes(length=2, byteorder="big")
    assert b[0] == 1
    assert b[1] == 2


main()
