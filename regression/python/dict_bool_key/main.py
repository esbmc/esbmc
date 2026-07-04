# KNOWNBUG: a dict with boolean keys mis-resolves subscript lookups. The
# bool key is stored/looked up with a type_id/size that does not match, so
# d[True]/d[False] fail even though CPython resolves them (and bool keys are
# hashable: {True: 1, False: 0}). Bool as a dict *value* works; only bool as
# a *key* is affected — the dict/set key type-tagging path in the list OM.
d = {True: 1, False: 0}
assert d[True] == 1
assert d[False] == 0
