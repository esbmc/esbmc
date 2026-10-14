from modstub import C

# Named import (not `import *`): only C is bound here, but C.check still
# resolves the module global TAG against modstub at call time (CPython LEGB).
obj: C = C(1)
obj.check()
