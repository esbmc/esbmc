# Dynamic (non-constant) tuple subscript: exercises the if-chain select path in
# tuple_handler::handle_tuple_subscript, including negative-index normalisation.
def main() -> None:
    t = (10, 20, 30)
    i = 1
    assert t[i] == 20
    assert t[-1] == 30
    assert t[i - 1] == 10


main()
