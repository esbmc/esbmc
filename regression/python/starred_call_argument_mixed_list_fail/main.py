def pick(arr, k):
    return arr[k]


def check(input_data, expected):
    assert pick(*input_data) == expected


def main():
    check([[3, 4, 5], 1], 4)
    check([[7, 8], 0], 8)


main()
