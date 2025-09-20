numbers = [1, 2.5, False, "Abc"]

assert(numbers[0] == 1)
assert(numbers[1] == 2.5)
assert(numbers[2] == False)
assert(numbers[3] == "Abc")

numbers.append(4)
assert(numbers[4] == 4)
