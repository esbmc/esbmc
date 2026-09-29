# A string subscript is tagged #cpp_type == "char" so that
# get_python_type_category tells a 1-char string element from an 8-bit int. The
# tag has to survive the symbol-table seam: passing the element through a
# variable stores its type, and without the spelling the comparison folds
# cross-type -- Eq to False, NotEq to True (esbmc/esbmc#4715).
val = "hello"[0]
assert val == "h"
assert val != "x"
