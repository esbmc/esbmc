def search(x:int, seq:list[int]):
    """ Takes in a value x and a sorted sequence seq, and returns the position that x should go to such that the sequence remains sorted """
    for i in range(len(seq)):
        if x <= seq[i]:
            return i
    return len(seq)

assert search(5, [1, 10, 2]) == 1
