# A starred parameter holding a list and an int fills a list parameter and an
# int parameter; each element is read at a constant index. KNOWNBUG until
# #8534 types a list parameter per element (#8472).
def pick(arr, k):
    return arr[k]


def check(input_data, expected):
    assert pick(*input_data) == expected


def main():
    check([[3, 4, 5], 1], 4)
    check([[7, 8], 0], 7)


main()
