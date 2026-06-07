# Negative: a wrong expected list must yield a real VERIFICATION FAILED, proving
# the result is genuinely computed (not a vacuous/abort verdict).
def derivative(xs: list):
    return [(i * x) for i, x in enumerate(xs)][1:]


assert derivative([3, 1, 2]) == [1, 99]
