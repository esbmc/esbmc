# #9161: the program never imports importlib.machinery, so CPython raises
# AttributeError evaluating this base.
import importlib


class C(importlib.machinery.ModuleSpec):
    pass


assert 1 == 1
