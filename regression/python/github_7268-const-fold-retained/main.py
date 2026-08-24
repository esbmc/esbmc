# The counterweight to github_7268-const-fold-keeps-call: outside a
# reachability run the fold is a wanted optimisation, so it must stay.
def pick(x: int) -> int:
    if x > 0:
        return 1
    return 0


def main():
    assert pick(5) == 1


main()
