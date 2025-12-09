numbers = [10, 20, 30, 40, 50]

# Test various negative indices
assert numbers[-1] == 50    # Last element
assert numbers[-5] == 10    # First element via negative index
assert numbers[-3] == 30    # Middle element
assert numbers[-2] == 40    # Second to last

# Test negative indexing with strings
names = ['Alice', 'Bob', 'Charlie']
assert names[-1] == 'Charlie'
assert names[-3] == 'Alice'

empty = []
single = ['only']

# Test empty list properties
assert len(empty) == 0
assert empty != single

# Test single element list
assert len(single) == 1
assert single[0] == 'only'
assert single[-1] == 'only'

special_chars = ['α', 'β', 'γ', '🚀', '★']

# Test Unicode character handling
assert special_chars[0] == 'α'
assert special_chars[3] == '🚀'
assert special_chars[-1] == '★'


words = ['hello', 'world', 'test']
target = 'hello'

# Test direct string comparison with array elements
assert words[0] == target
assert words[0] == 'hello'
assert words[1] != 'hello'

# Test with different string lengths
short = ['hi']
long_word = 'hello'
assert short[0] != long_word


