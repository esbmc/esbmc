# #9126: the second comprehension rebinds `x` to an int after the first bound
# it to a str.
arr = [int(x) for x in ["1", "2"]]
arr2 = [x for x in arr]
assert arr2[0] == 1 and arr2[1] == 2
