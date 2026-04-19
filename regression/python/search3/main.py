def search(x, seq) -> int:
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

assert search(40, [10, 20, 30]) == 3
