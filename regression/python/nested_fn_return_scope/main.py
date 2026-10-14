# A function's return type must be inferred from *its own* return statements,
# not from a nested helper's. Here `classify` returns a str, while the nested
# `has_open` returns bool. If the nested bool returns leak into classify's
# inferred type, classify is mis-typed as bool and the str `==` comparison
# below is wrongly folded to a constant (cross-type fold) instead of running.
# Regression guard for GitHub #4807 (humaneval_119 / humaneval_127 family).


def classify(s):
    def has_open(t):
        for c in t:
            if c == '(':
                return True
        return False

    return 'open' if has_open(s) else 'none'


assert classify('(x)') == 'open'
assert classify('xyz') == 'none'
