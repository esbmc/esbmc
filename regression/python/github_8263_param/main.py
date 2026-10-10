# An unannotated parameter is void* too, and its call sites fix its type, so
# arithmetic on it must still convert and verify (#8263).
def total(a, b):
    return a + b


def main():
    assert total(2, 3) == 5


main()
