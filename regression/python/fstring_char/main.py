# f-string {:c} renders an integer as the character with that code point, for a
# constant integer in the printable-through-DEL ASCII range (1-127).
assert f"{65:c}" == "A"
assert f"{97:c}" == "a"
assert f"{48:c}" == "0"
n = 90
assert f"{n:c}" == "Z"
