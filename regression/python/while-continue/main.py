def sum_even_numbers(n):
    total = 0
    i = 1
    while i <= n:
        if i % 2 != 0:
            i = i + 1
            continue
        total = total + i
        i = i + 1
    return total


result = sum_even_numbers(10)

assert result == 30
