# Negative variant of github_5464: the dict has three keys, so len(sorted(m))
# is 3 and the assertion that it equals 2 must fail.
def main() -> None:
    m = {3: 30, 1: 10, 2: 20}
    assert len(sorted(m)) == 2


main()
