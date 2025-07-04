data = [(1, 50), (2, 30), (3, 40)]
sorted_data = sorted(data, key=lambda x: x[1])

# Ensure list is sorted by the second value in each tuple
assert sorted_data == sorted(data, key=lambda x: x[1]), "Tuples are not sorted correctly!"
