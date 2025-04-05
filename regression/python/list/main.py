l1 = [1,2,3]
assert l1[0] == 1
assert l1[1] == 2
assert l1[2] == 3

l2 = [True, False]
assert l2[0] == True
assert l2[1] == False

l3 = [1.2, 2.5]
assert l3[0] == 1.2
assert l3[1] == 2.5

l4 = ["abc", "def"]
assert l4[0] == "abc"
assert l4[1] == "def"

def make_list() -> list[int]:
    l5 = [1,2,3]
    return l5

l6 = make_list()
assert l6[0] == 1
assert l6[1] == 2
assert l6[2] == 3


def foo(l: list[int]) -> list[int]:
    return l

l7 = [2,2,3]
l7[0] = 1
assert l7[0] == 1
assert l7[1] == 2
assert l7[2] == 3