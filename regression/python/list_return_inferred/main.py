# An unannotated function that returns a sliced list comprehension must infer a
# list return type, not default to scalar double. Defaulting to double retyped
# the returned list's elements and broke equality against an int list
# (regression/humaneval/humaneval_62).
def derivative(xs: list):
    return [(i * x) for i, x in enumerate(xs)][1:]


assert derivative([3, 1, 2]) == [1, 4]
