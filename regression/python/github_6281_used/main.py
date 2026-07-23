# github.com/esbmc/esbmc/issues/6281
# Relative import whose name is *used* (called). This exercises the annotator's
# get_function_return_type / wildcard-import paths, which read the ImportFrom
# "module" field. For `from . import X` that field is null; the earlier fix only
# guarded the converter, so these annotator sites still core-dumped with an
# uncatchable nlohmann type_error. The name stays unresolved, so ESBMC must
# degrade to a NameError instead of aborting.
from . import helper


def main() -> None:
    x = helper()
    assert x == 42


main()
