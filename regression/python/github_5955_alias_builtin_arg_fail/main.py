# min([l]) returns l itself: a container-literal argument smuggles the
# reference past the pure-builtin exemption (GitHub #5955 review F3).
l = [1, 2]
m = min([l])
m.append(3)
assert l.count(3) == 0
