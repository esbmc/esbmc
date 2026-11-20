# #10143: [] or 0 is 0, but 0 or [] is [], which a void * result reads as true.
xs = []
z = 0 or xs
assert not z
