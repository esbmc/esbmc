# github.com/esbmc/esbmc/issues/6264
# A str reached through a list subscript (list-of-str element) has no .append;
# it must raise AttributeError, not be routed into the list model.
def main() -> None:
    xs = ["a", "b"]
    xs[0].append("d")


main()
