# Regression for #7434: the branching waypoints ESBMC took inside the list
# operational model were emitted with this file's name and the model's line
# numbers, pointing a validator past the end of a ten-line file. A waypoint
# that cannot name an input file is dropped instead.
def main() -> None:
    xs: list[int] = []
    xs.append(7)
    assert xs[0] == 8


main()
