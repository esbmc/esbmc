# f-string integer base presentations x/X/o/b now fold for a constant integer
# (they were previously unsupported, unlike the format()/str.format() paths).
assert f"{255:x}" == "ff"
assert f"{255:X}" == "FF"
assert f"{8:o}" == "10"
assert f"{5:b}" == "101"
assert f"{10:b}" == "1010"
assert f"{0:x}" == "0"
assert f"{-255:x}" == "-ff"
n = 255
assert f"{n:x}" == "ff"
