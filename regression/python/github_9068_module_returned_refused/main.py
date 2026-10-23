# The same as github_9068_imported_returned_refused, through the module name.
import mod

h = mod.pick(True)
b = h()
assert b == "a"
