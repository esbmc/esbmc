# KNOWNBUG: min()/max() over a string should compare its characters
# (min("banana") == "a", max("banana") == "n"), but ESBMC returns the wrong
# result. min()/max() over a list or tuple works; a string argument is not
# routed through the character-iterating comparison in handle_min_max.
assert min("banana") == "a"
assert max("banana") == "n"
