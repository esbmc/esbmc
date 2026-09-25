# Exercises the --python-irep2-adjust-only branch/loop condition bool cast.
# `if n:` and `while i:` keep the raw signedbv condition in the converter;
# clang_c_adjust casts it (gen_typecast_bool). Without the cast the guard reaches
# the SMT layer as a bitvector where a Boolean is required and bitwuzla rejects
# it with "term with unexpected sort at index 0".
def classify(n: int) -> int:
    total = 0
    if n:
        total = 1

    i = 2
    while i:
        total = total + 1
        i = i - 1

    return total


assert classify(5) == 3
assert classify(0) == 2
