def search(x:int, seq:list[int]):
    for i in range(len(seq)):
        if x <= seq[i]:
            return i
    return len(seq)

def main():
    # Example sequences
    seq1 = [1, 3, 5, 7]
    seq2 = [2, 4, 6, 8]
    
    # Test cases with asserts
    assert search(0, seq1) == 0
    assert search(4, seq1) == 2
    assert search(10, seq1) == 4
    
    assert search(2, seq2) == 0
    assert search(5, seq2) == 2
    assert search(9, seq2) == 1

if __name__ == "__main__":
    main()
