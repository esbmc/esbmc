def main():
    apply = lambda g, v: g(v)
    inc = lambda x: x + 1
    assert apply(inc, 5) == 6


main()
