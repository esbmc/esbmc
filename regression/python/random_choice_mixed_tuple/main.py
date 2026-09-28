# The inline tuple fold builds one conditional, which carries one type, so a
# tuple whose members differ in type would coerce every result to the first
# member's type and silently mis-answer later comparisons. Report it instead of
# emitting a claim. CPython accepts this call, so a verdict here would be a
# false one either way (#7673).
import random


def main() -> None:
    c = random.choice((1, "two"))
    assert c == 1 or c == "two"


main()
