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
l7[0] = 1    # Modify element after initialisation
assert l7[0] == 1
assert l7[1] == 2
assert l7[2] == 3


def func1(x:float) -> float:
    y = x
    return 2.0

def func2(l: list[float]) -> None:
    l[0] = func1(l[0]) # Updating list element within a function

l8 = [1.0]
func2(l8)
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 9216cc719 ([python-frontend] Handling array literals (#2467))
assert l8[0] == 2.0

def bar(x:list[int]) -> None:
    assert x[0] == 1

x = 0    
<<<<<<< HEAD
bar([1,2,3])
=======
assert l8[0] == 2.0
>>>>>>> f59fd87f9 ([python-frontend] Reuse C libm models for NumPy math functions (#2395))
=======
bar([1,2,3])
>>>>>>> 9216cc719 ([python-frontend] Handling array literals (#2467))
