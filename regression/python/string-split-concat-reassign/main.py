def main() -> None:
    # Receiver reassigned before split(): the constant fold resolves the FIRST
    # binding ("a."+"b") instead of "p.q.r", a pre-existing get_var_value
    # limitation noted in string_handler. CPython splits "p.q.r" -> 3 parts.
    s = "a." + "b"
    s = "p.q.r"
    parts = s.split(".")
    assert len(parts) == 3


main()
