# Step-1 list slice assignment under --incremental-bmc. This used to be
# rejected with "List slice assignment (a[i:j] = ...) is not supported";
# it is now lowered to the __ESBMC_list_slice_assign model, which mutates
# the list in place with CPython semantics (a becomes [1, 9, 4]).
a = [1, 2, 3, 4]
a[1:3] = [9]
assert a == [1, 9, 4]
