# Negative counterpart: 258 big-endian is b'\x01\x02', so b[0] is 1, not 2. The
# hop-off must reach the solver and report the violation rather than aborting in
# migrate_expr on an unresolved shift.
def main() -> None:
    b = (258).to_bytes(length=2, byteorder="big")
    assert b[0] == 2


main()
