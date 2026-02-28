import ll

f = ll.Foo()

# Test edge cases:
# 1. Mixed positional and keyword arguments
# 2. Keyword argument overriding default value
# 3. Partial keyword arguments (some use defaults, some provided)
f.test(5, z="provided", w=100)
