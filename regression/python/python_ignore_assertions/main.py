# --python-ignore-assertions drops the program's own assert but keeps it as an
# assumption, so the run is not cut short there and l[i] stays in bounds.
def main():
    l = [0, 1]
    i = nondet_int()
    assert 0 <= i and i < 2
    x = l[i]
    if i == 1:
        raise AssertionError


main()
