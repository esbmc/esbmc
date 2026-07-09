# Boolean membership in a set now resolves correctly. Sets share the list-OM
# key type-tagging path with dict keys; the bool element was stored widened
# to a long while lookups queried at bool's 1-byte size.
s = {True, False}
assert True in s
assert False in s
t = set([True])
assert True in t
