values = [1, 2, 3, 4, 5, 6]
odds = [x for x in values if x % 2 == 1]

assert odds == [1, 3, 5]

