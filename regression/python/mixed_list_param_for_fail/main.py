# Negative variant of mixed_list_param_for_eq: the heterogeneous list elements
# are read at their true values, so a wrong expected result must be reported as a
# violated assertion (esbmc/esbmc#5156).
def grade(xs):
    out = []
    for x in xs:
        if x == 4.0:
            out.append("A")
        elif x > 1.5:
            out.append("B")
        else:
            out.append("C")
    return out


if __name__ == "__main__":
    assert grade([4.0, 3, 1.7]) == ["A", "B", "C"]
