lst = [1, 2, 3]

lst.append(4)
lst.insert(2, 99)
assert lst == [1, 2, 99, 3, 4, 5, 6]
