# #9204: the program never imports importlib.machinery, so CPython raises
# AttributeError evaluating this annotation.
import importlib


def f(x: importlib.machinery.ModuleSpec) -> int:
    return 1


assert f(None) == 1
