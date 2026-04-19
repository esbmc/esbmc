def search(x:int, seq:list[int]) -> int:
    for i, elem in enumerate(seq):
        if x <= elem:
            return i
    return len(seq)

x = 2
seq = [1, 2, 3, 4]
idx = search(x, seq)
assert idx == 0
