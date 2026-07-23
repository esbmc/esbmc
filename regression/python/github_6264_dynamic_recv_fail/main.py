# github.com/esbmc/esbmc/issues/6264 (review) / #5955
# Negative twin of github_6264_dynamic_recv: alias.append routes into the list
# model and mutates xs, so the "unchanged" expectation is violated.
xs = [1, 2]
alias = min([xs])
alias.append(3)
assert xs.count(3) == 0
