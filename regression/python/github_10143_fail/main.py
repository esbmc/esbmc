# #10143: with xs empty, (xs and True) is false and the loop never runs.
xs = []
k = 0
while (xs and True) and k < 1:
    k += 1
assert k == 1
