# #10143: "" and True is "", which is false.
z = "" and True
assert z
