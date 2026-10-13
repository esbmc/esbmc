# --no-assertions drops the assert but keeps its assume. The inductive step
# cannot bound j, so only the forward condition can prove the loop ends, and
# it needs that assume.
def main():
    l = [0, 1]
    i = nondet_int()
    assert 0 <= i and i < 2
    j = 0
    while j < i:
        j += 1
    x = l[j]


main()
