# Unlike --no-assertions, --python-ignore-assertions keeps the operational
# model's own checks: dividing a value that may hold a str still raises the
# model's TypeError, while the program's assert is only assumed.
def main():
    if nondet_bool():
        r = 1
    else:
        r = "s"
    assert r is not None
    h = r / 2


main()
