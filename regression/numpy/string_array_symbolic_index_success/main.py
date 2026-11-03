import numpy as np

a = np.array(["ab", "cd", "ef"])
k = np.random.randint(1, 3)
s = a[k]

assert s == "cd" or s == "ef"
assert s[1] == "d" or s[1] == "f"
