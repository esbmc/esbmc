# f's return type diverges (int vs str) across an if/else branch, based on a
# nondet condition -- the function-return analogue of a dynamically-typed
# local variable.

def f(cond):
    if cond:
        return 1
    else:
        return "a"

x = f(nondet_bool())
assert x == 1 or x == "a"
