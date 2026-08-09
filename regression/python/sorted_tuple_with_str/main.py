# sorted() over tuples carrying a string falls through the constant-integer
# tuple fold onto the runtime tuple-sort model, which retypes elements as int
# (see fold_sorted_constant_tuples in function_call/expr.cpp). Indexing the
# result then reports "'int' object is not subscriptable". The same list
# unsorted indexes fine, and int-only tuples sort fine -- see
# sorted_tuple_ints.

v = sorted([(2, "b"), (1, "a")])
assert v[0][0] == 1
