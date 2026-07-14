# Negative variant of github_5495: the formatted string is "3-4", so asserting
# it equals "3-5" must fail.
def main() -> None:
    s = "%d-%d" % (3, 4)
    assert s == "3-5"


main()
