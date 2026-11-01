# github issue 9355: the entry script's AST was overwritten by the model of the
# same name, so this program was refused (or, for string.py, never verified).
x: int = 1
try:
    raise ValueError("v")
except ValueError:
    x = 2
assert x == 2
