def search(x, seq:list[int]):
    """ Takes in a value x and a sorted sequence seq, and returns the
    position that x should go to such that the sequence remains sorted """
    output = 0
    while output < len(seq):
        if x > seq[output]:
            output += 1
        else:
            break
    return output

# Test case 1: Empty sequence
result = search(5, [])
assert result == 0

# Test case 2: Insert at beginning
result = search(1, [2, 3, 4, 5])
assert result == 0    

# Test case 3: Insert at end
result = search(6, [1, 2, 3, 4])
assert result == 4

# Test case 4: Insert in middle
result = search(3, [1, 2, 4, 5])
assert result == 2

# Test case 5: Single element sequence - insert before
result = search(1, [2])
assert result == 0

# Test case 6: Single element sequence - insert after
result = search(3, [2])
assert result == 1

# Test case 7: Element already exists (should insert after existing)
result = search(3, [1, 2, 3, 4, 5])
assert result == 2

# Test case 8: Multiple duplicates
result = search(3, [1, 2, 3, 3, 3, 4])
assert result == 3 # this should fail

