def next_palindrome(digit_list):
    high_mid = len(digit_list) // 2
    low_mid = (len(digit_list) - 1) // 2
    while high_mid < len(digit_list) and low_mid >= 0:
        if digit_list[high_mid] == 9:
            digit_list[high_mid] = 0
            digit_list[low_mid] = 0
            high_mid += 1
            low_mid -= 1
        else:
            digit_list[high_mid] += 1
            if low_mid != high_mid:
                digit_list[low_mid] += 1
            return digit_list
    return [1] + (len(digit_list) - 1) * [0] + [1]

assert next_palindrome([1, 4, 9, 4, 1]) == [1, 5, 0, 5, 1]
assert next_palindrome([1, 3, 1]) == [1, 3, 1] # should fail
# assert next_palindrome([4, 7, 2, 5, 5, 2, 7, 4]) == [4, 7, 2, 6, 6, 2, 7, 4]
# assert next_palindrome([4, 7, 2, 5, 2, 7, 4]) == [4, 7, 2, 6, 2, 7, 4]
# assert next_palindrome([9, 9, 9]) == [1, 0, 0, 1]

