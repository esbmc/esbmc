# github.com/esbmc/esbmc/issues/6281
# `from . import X` (relative import, no module name) used to crash ESBMC with an
# uncaught nlohmann type_error. It must degrade to an unresolved import and let
# the rest of the module verify.
from . import helper


def main() -> None:
    total = 0
    for i in range(5):
        total += i
    assert total == 10


main()
