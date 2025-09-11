def search(x, seq):
    for i in range(len(seq)):
        if x <= seq[i]:
            return i
    return len(seq)


assert search(42, [-5, 1, 3, 5, 7, 10]) == 6
