# A break that targets a loop *nested inside* the try does not escape the try,
# so it must NOT be refused: the loop captures the break and the finally still
# runs on the fall-through path. Pins the in-loop boundary of the escape check.
x = 0
try:
    for i in range(3):
        if i == 1:
            break
    x = 1
finally:
    x = x + 10

assert x == 11
