# Out-of-range dynamic tuple index must trip the bounds assertion built in
# handle_tuple_subscript (0 <= idx_norm < n).
def main() -> None:
    t = (10, 20, 30)
    i = 5
    assert t[i] == 30


main()
