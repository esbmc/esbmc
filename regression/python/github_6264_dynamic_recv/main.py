# github.com/esbmc/esbmc/issues/6264 (review) / #5955
# min([xs]) is typed int by the annotator though it aliases xs itself. The
# alias.append must still route into the list model (a pure resolves-to-list
# positive check drops it, since the alias is not statically a list), so the
# append reaches xs and count sees the new element. Guards that regression.
xs = [1, 2]
alias = min([xs])
alias.append(3)
assert xs.count(3) == 1
