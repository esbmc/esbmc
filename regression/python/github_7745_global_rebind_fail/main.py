# A global write in a function keeps the parameter untyped (#7745).
x = 1
def set_x():
    global x
    x = 2.5
set_x()
g = lambda n: n + 0
assert g(x) == 2
