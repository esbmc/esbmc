# Regression for #5495: `str % args` (printf-style formatting) was mis-lowered
# as numeric modulo on the format string's pointer, crashing the SMT backend
# (SIGSEGV) when the result was consumed by len()/==. Constant format + constant
# args are now folded to the correct string literal. The `==` checks validate
# exact content (and therefore length); len() is exercised via a bound variable.
def main() -> None:
    assert "%d" % 5 == "5"
    assert "%i" % 7 == "7"
    assert "%d-%d" % (3, 4) == "3-4"
    assert "%s" % "hi" == "hi"
    assert "val=%s" % 42 == "val=42"
    assert "%s" % True == "True"
    assert "%x" % 255 == "ff"
    assert "%X" % 255 == "FF"
    assert "%o" % 8 == "10"
    assert "%c" % 65 == "A"
    assert "%d%%" % 50 == "50%"
    assert "%d" % (-7) == "-7"
    s = "%d" % 5
    assert len(s) == 1


main()
