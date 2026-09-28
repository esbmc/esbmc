# An unannotated parameter whose body uses it as a list is refined to the list
# model. That refinement was limited to the entry file, so the same function in
# an imported module kept an Any parameter and len() over it ran strlen.
from helper import size, first, appended

assert size([1, 2, 3]) == 3
assert first([5, 7]) == 5

xs = [1, 2]
assert appended(xs, 9) == 3
assert xs[2] == 9
