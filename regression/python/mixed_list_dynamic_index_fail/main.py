# Negative variant of mixed_list_dynamic_index: the dynamic-index read returns
# the real computed values (not a constant), so a wrong expected list must be
# reported as a violated assertion.
def tri(n):
    my_tri = [1, 3, 2.0]
    for i in range(3, n + 1):
        my_tri.append(my_tri[i - 1] + my_tri[i - 2] + (i + 3) / 2)
    return my_tri


if __name__ == "__main__":
    assert tri(3) == [1, 3, 2.0, 9.0]
