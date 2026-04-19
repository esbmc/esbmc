nil = 0
num = 0
max = 1
cap = 'A'
low = 'a'

print('Equality: \t', nil, '==', num, nil == num)
assert nil == num

print('Equality: \t', cap, '==', low, cap == low)
assert not (cap == low)
