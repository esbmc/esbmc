# Same as dynamic_type_function_return_module_call_dedup, but asserts the
# list length a double-pop would produce, so a regression here shows up as
# an unexpected VERIFICATION SUCCESSFUL rather than a silent pass.
l = [1, 2, 3]
x = l.pop()
assert x == 3
assert len(l) == 1
