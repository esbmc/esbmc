# Exercises the --python-irep2-adjust-only implicit dereference of a pointer
# callee. A lambda read back out of a container has a callee that is a typecast
# of a member read, not a symbol, so the symbol-keyed wrapper never fired and the
# call went through the pointer value itself -- returning a corrupted double
# rather than failing outright.
def pick(key: str) -> float:
    return {'+': lambda: 1.0, '-': lambda: 2.0}[key]()


assert pick('+') == 1.0
assert pick('-') == 2.0
