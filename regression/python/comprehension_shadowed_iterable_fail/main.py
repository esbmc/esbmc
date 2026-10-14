def main():
    xs = [1, 2, 3]
    node = xs
    out = [node for node in node]
    assert len(out) == 2


main()
