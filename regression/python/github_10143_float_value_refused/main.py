# #10143: 0.5 or 2 is 0.5, which a void * result cannot hold.
z = 0.5 or 2
assert z == 0.5
