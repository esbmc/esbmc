# github #9447: `"" or "xy"` is "xy", whose first character is "x".
e = "" or "xy"
assert e[0] == "y"
