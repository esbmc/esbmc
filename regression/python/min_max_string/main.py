# min()/max() over a string return the extreme character (min("cba") == "a").
# Previously the single string-iterable form fell to the runtime model, which
# only handles list/tuple iterables, and gave a wrong result. Now folded over
# the constant string's characters (literal and constant-variable forms).
assert min("cba") == "a"
assert max("cba") == "c"
assert min("z") == "z"
assert min("31524") == "1"
assert max("aAzZ") == "z"
s = "dcba"
assert min(s) == "a"
assert max(s) == "d"
