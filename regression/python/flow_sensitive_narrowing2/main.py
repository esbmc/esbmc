def normalize(x):
    first = ""
    second = [""]
    third = 0
    fourth = ""

    i = 0

    while i < 4:
        if i == 0:
            if isinstance(x, int):
                first = str(x)

        elif i == 1:
            if isinstance(first, str):
                second = [first]

        elif i == 2:
            if isinstance(second, list):
                third = len(second)

        else:
            if isinstance(third, int):
                fourth = str(third)

        i += 1

    assert isinstance(first, str)
    assert isinstance(second, list)
    assert isinstance(third, int)
    assert isinstance(fourth, str)

    strings = []
    numbers = []
    lists = []

    strings.append(first)
    strings.append(fourth)
    numbers.append(third)
    lists.append(second)

    assert isinstance(strings[0], str)
    assert isinstance(strings[1], str)
    assert isinstance(numbers[0], int)
    assert isinstance(lists[0], list)

    return first


value = normalize(5)
assert value == "5"
