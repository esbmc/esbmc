def main() -> None:
    # Negative: reverse=True puts 3 first, not 1.
    xs = [3, 1, 2]
    xs.sort(reverse=True)
    assert xs[0] == 1
main()
