nums = [0, 1, 2]
nested = [y for x in nums for y in range(x)]

assert nested == [0, 0, 1]