# Companion to str_unannotated_param: the presence of an unannotated-param
# str() function (which falls back to a nondet string) must not crash conversion
# nor mask a genuine assertion failure elsewhere. The false assertion below must
# still be reported as FAILED.


def stringify(x):
    return str(x)


if __name__ == "__main__":
    n = 1
    assert n == 2
