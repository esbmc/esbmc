def main() -> None:
    parts = ".12".split(".")
    i = 0
    while i < len(parts):
        if parts[i] == "":
            assert False
        i += 1


main()
