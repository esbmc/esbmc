
def sqrt(x, epsilon):
    approx = x / 2
    while abs(x - approx ** 2) > epsilon:
        approx = 0.5 * (approx + x / approx)
    return approx

"""
def sqrt(x, epsilon):
    approx = x / 2
    while abs(x - approx * approx) > epsilon:
        approx = 0.5 * (approx + x / approx)
    return approx
"""

assert sqrt(2, 0.01) == 1.4166666666666665
assert sqrt(2, 0.5) == 1.5
assert sqrt(2, 0.3) == 1.5
assert sqrt(4, 0.2) == 2
#assert sqrt(27, 0.01) == 5.196164639727311
#assert sqrt(33, 0.05) == 5.744627526262464
assert sqrt(170, 0.03) == 13.038404876679632