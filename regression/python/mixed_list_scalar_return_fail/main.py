# Negative variant of mixed_list_scalar_return: same heterogeneous list/scalar
# function, but an assertion that does not hold on the list branch. Confirms the
# recovered list value flows precisely (a false claim is refuted, not nondet or
# vacuously true) — i.e. the fix is sound, not merely non-crashing.


def f(txt):
    if " " in txt:
        return txt.split()
    else:
        return len(txt)


assert f("Hello world!") == ["WRONG", "VALUE"]
