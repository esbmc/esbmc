# Binary tree: multiple None-initialised self-referential fields in the
# same class. Exercises per-attribute type inference (both .left and
# .right should each be inferred as Tree*, independently).

class Tree:
    def __init__(self, v):
        self.value = v
        self.left = None
        self.right = None


root = Tree(1)
lchild = Tree(2)
rchild = Tree(3)

root.left = lchild
root.right = rchild

assert root.left.value == 2
assert root.right.value == 3
