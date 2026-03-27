# Regression test for GitHub issue #3808:
# Recursive generator functions crash ESBMC with:
#   "Assertion failed: (is_array_type(source) || is_vector_type(source)),
#    function index2t"
#
# Root cause: the preprocessor emitted `ESBMC_iter: Any = flatten(x)` which
# made the loop-variable void*, and subscripting void* fails the index2t
# assertion.  Fix: _transform_recursive_generator annotates the iterable
# parameter as list[Any] and emits `ESBMC_iter: list[Any] = flatten(x)`.


def flatten(arr):
    for x in arr:
        if isinstance(x, list):
            for y in flatten(x):
                yield y
        else:
            yield x


# Flat list: no recursion needed
result = list(flatten([1]))
assert len(result) == 1

# One level of nesting
result2 = list(flatten([[1, 2]]))
assert len(result2) == 2
