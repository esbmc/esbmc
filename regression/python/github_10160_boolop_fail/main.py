# xs.append(0) is None, so the branch is not taken (#10160).
xs = [1]
k = 0
if xs.append(0) and k == 0:
    k = 5
assert k == 5
