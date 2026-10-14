# Read-back of a dict built by a comprehension, inside a function whose
# return type does not flow into the subscript read. Previously the value
# type of a comprehension-built dict could not be resolved, so d[k] fell
# through to the char* default and returned the raw PyObject value pointer
# instead of the stored scalar -- a wrong verdict. Regression for #5222
# (the read-back portion, reproducer 2: list-iterable comprehension).


def from_list(lst):
    count = {i: 0 for i in lst}
    return count[5]


assert from_list([5]) == 0
