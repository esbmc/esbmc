# Comparing a call result inline looked the operand up in the symbol table,
# found nothing, and dereferenced the null: SIGSEGV, not a verdict (#7555).
# The FAILED verdict is itself a false alarm -- zip's elements are not modelled,
# and CPython holds this assertion. What this pins is that a verdict is reached
# at all, which no output line does on the crashing build.
def main() -> None:
    assert list(zip()) == []


main()
