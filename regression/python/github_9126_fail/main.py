# #9126
arr = [int(x) for x in ["1", "2"]]
arr2 = [x for x in arr]
assert arr2[1] == 1
