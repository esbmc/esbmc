# #10428: the copy used to read the first float stored in any list.
x = [5.0]
b = [1.0] + []
assert b[0] == 5.0
