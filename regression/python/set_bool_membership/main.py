# KNOWNBUG: membership of a boolean element in a set is mis-resolved. Sets
# share the list-OM key type-tagging path with dict keys; a bool element is
# stored/matched with a type_id/size that does not line up, so `True in s`
# fails even though CPython holds. Bool in a *list* works; only the set/dict
# key path is affected.
s = {True, False}
assert True in s
assert False in s
