# The filter is really applied: 10 is the unfiltered sum, which is what the
# dropped-filter lowering used to produce.
def main() -> None:
    xs = [1, 2, 3, 4]
    assert sum(x for x in xs if x > 2) == 10


main()
