# The for target rebinds l without an assignment statement (GitHub #5955).
l = [1, 2]
for l in [[9, 9]]:
    pass
assert l.count(9) == 0
