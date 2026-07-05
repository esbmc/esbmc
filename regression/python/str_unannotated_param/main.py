# str() of an unannotated parameter, modelled as Any (void*), has no statically
# stringifiable type. It must fall back to a sound nondet string rather than
# aborting conversion of the whole program. Here stringify() is never called, so
# its body is dead and verification of the rest of the program still completes.


def stringify(x):
    return str(x)


if __name__ == "__main__":
    assert True
