def main() -> None:
    s = nondet_str()
    idx = s.rfind("")

    assert idx == 0


main()
