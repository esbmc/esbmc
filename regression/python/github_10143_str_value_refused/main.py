# #10143: "" and True is "", which a void * result reads as true.
z = "" and True
assert not z
