basket = ['Apple']

basket.insert(10, 'Mango')  # index > len(list), should append
assert basket == ['Apple', 'Mango']
