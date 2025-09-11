lst = [1, 2, 3]

lst.append(4)
assert lst == [1, 2, 3, 4]

lst.extend([5, 6])
assert lst == [1, 2, 3, 4, 5, 6]

lst.insert(2, 99)
assert lst == [1, 2, 99, 3, 4, 5, 6]

lst.remove(99)
assert lst == [1, 2, 3, 4, 5, 6]

last_element = lst.pop()
assert last_element == 6
assert lst == [1, 2, 3, 4, 5]

index_of_3 = lst.index(3)
assert index_of_3 == 2

count_of_2 = lst.count(2)
assert count_of_2 == 1

lst.append(0)
lst.sort()
assert lst == [0, 1, 2, 3, 4, 5]

lst.reverse()
assert lst == [5, 4, 3, 2, 1, 0]
