word = [1, 2, 3, 4, 5]
even_count = 0
odd_count = 0
for item in word:
    if item % 2 == 0:
        even_count = even_count + 1
    else:
        odd_count = odd_count + 1
# assert even_count == 2
# assert odd_count == 3
