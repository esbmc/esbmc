g = 1


def bump() -> int:
    global g
    g = 10
    return 5


flag = True
assert flag and g + bump() == 6

g = 1
flag = False
assert flag or g + bump() == 6
assert g == 10

g = 1
assert not (flag and g + bump() == 6)
assert g == 1

lst = [1, 2]
i = 5
assert not (i < 2 and lst[i] + bump() == 3)
assert g == 1
