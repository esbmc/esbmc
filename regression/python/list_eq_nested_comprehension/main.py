# KNOWNBUG: equality of two lists-of-lists fails when the inner lists were
# built by DIFFERENT means (here a comprehension on the left vs a literal on
# the right). Both sides are structurally list[list[int]] and CPython holds.
#
# Root cause: __ESBMC_list_eq detects a nested-list element only when its
# element type_id equals the single list_type_id passed from the == call site
# (hash of the left operand's type string, list_query.cpp:491). A
# comprehension-built inner list and a literal inner list are given different
# type-representation strings by the frontend, so their element type_ids
# differ; the comparator then byte-compares the two inner-list *pointers*
# (list.c:330,359) instead of recursing, and reports inequality.
#
# The companion test list_eq_nested_same shows the same value compares equal
# when both sides are built the same way. Fixing this needs the frontend to
# give structurally-identical nested lists the same type representation.
m = [[i * j for j in range(2)] for i in range(2)]
assert m == [[0, 0], [0, 1]]
