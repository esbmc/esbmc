# github.com/esbmc/esbmc/issues/6265
# list + non-list is a TypeError in Python (only list + list concatenates).
# Uncaught -> VERIFICATION FAILED.
def main() -> None:
    xs = [1, 2]
    ys = xs + 3


main()
