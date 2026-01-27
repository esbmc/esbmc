t = (1, 2, 3)
# This should trigger a frontend error or verification failure
t[0] = 10 
assert t[0] == 1
