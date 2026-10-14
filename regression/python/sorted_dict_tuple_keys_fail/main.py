# The keys are really sorted and read, not assumed to be in insertion order.
def main() -> None:
    d = {(3, 1): 10, (1, 2): 20}
    for edge in sorted(d):
        u, v = edge
        assert u == 3


main()
