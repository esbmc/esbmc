# Comparing a call result inline looked the operand up in the symbol table,
# found nothing, and dereferenced the null: SIGSEGV, not a verdict (#7555).
# CPython holds this assertion, so the verdict is now checked, not just reached.
def main() -> None:
    assert list(zip()) == []


main()
