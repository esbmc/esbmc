def search(x:int, seq:list[int]) -> int:
    if len(seq) == 0:
        return 0
    if x < seq[0]:
        return 0
    elif x > seq[len(seq) - 1]:
        return len(seq)
    else:
        for i in range(len(seq) - 1):  # avoid index out of range
            if x > seq[i] and x <= seq[i + 1]:
                return i + 1

assert search(5, []) == 0
assert search(1, [10, 20, 30]) == 0
assert search(25, [10, 20, 30]) == 2
assert search(20, [10, 20, 30]) == 1
