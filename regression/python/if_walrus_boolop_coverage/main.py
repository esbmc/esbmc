# Coverage modes still accept a walrus in an `if` condition's and/or operand.
a = 1
if a and (x := a + 1):
    assert x == 2
