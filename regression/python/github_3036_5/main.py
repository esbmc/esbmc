def main() -> None:
    xs = [1, 2, 3]
    # Adding int to float should produce a list of floats
    ys = [x + 0.5 for x in xs]
    assert ys == [1.5, 2.5, 3.5]
    assert isinstance(ys[0], float) 

main()
