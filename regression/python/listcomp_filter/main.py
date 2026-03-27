values = [1, 2, 3]
odds = [x for x in values if x % 2 == 1]

assert odds == [1, 3]
