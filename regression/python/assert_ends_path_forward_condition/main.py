# --no-assertions drops the assert but keeps its assume, so l[i] stays in
# bounds and the forward condition must see that assume to prove the bound.
def main():
    l = [0, 1]
    i = nondet_int()
    assert 0 <= i and i < 2
    x = l[i]


main()
