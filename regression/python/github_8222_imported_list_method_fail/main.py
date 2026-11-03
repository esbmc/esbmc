# #8222: a method named like a list method on a class from another module
# runs the class's own method, not the list model.
from bag import Bag

b = Bag()
b.append(5)
assert b.items[2] == 5
