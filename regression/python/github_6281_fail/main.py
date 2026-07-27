# github.com/esbmc/esbmc/issues/6281
# Same relative import; the module must still convert and its assertions be
# checked (here a wrong one), rather than crashing at the import.
from .. import helper


def main() -> None:
    total = 0
    for i in range(5):
        total += i
    assert total == 11


main()
