# A return/break/continue that escapes a try with a finally is rejected: the
# current lowering appends the finally on the fall-through path, which such a
# jump would bypass. ESBMC refuses it rather than return an unsound verdict.
# (This is valid Python, so it runs cleanly under CPython.)
def h() -> int:
    try:
        return 1
    finally:
        pass


print(h())
