# The unsound direction of github_4715_cpp_type_char: without the #cpp_type
# spelling this folds to True and ESBMC proves a false property.
val = "hello"[0]
assert val != "h"
