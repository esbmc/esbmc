def main():
    xs = [10, 20, 30]
    i = 5
    try:
        t = xs[i]
        assert False
    except IndexError:
        assert True


if __name__ == "__main__":
    main()
