# The unchecked ValueError bounds i. The inductive step cannot bound j, so
# only the forward condition proves the loop ends.
def main() -> None:
    l = [0, 1]
    i = nondet_int()
    if i < 0 or i > 1:
        raise ValueError
    j = 0
    while j < i:
        j += 1
    x = l[j]


main()
