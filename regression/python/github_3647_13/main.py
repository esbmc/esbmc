def main():
    d: dict[bool, int] = {True: 1, False: 2}
    for k, v in d.items():
        assert isinstance(k, bool)

main()
