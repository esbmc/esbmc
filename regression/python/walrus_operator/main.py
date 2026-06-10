# PEP 572 walrus operator `:=` in the contexts ESBMC supports: an `if`
# condition, a standalone assignment expression, and a comprehension filter.
data = [1, 2, 3, 4]

# if-condition: the binding is read in the body
if (n := len(data)) > 2:
    assert n == 4

# standalone: the expression evaluates to the bound value
x = (y := 5)
assert x == 5 and y == 5

# comprehension filter: the binding is reused in the element
result = [d for v in data if (d := v * 2) > 4]
assert result == [6, 8]
