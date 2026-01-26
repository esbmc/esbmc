def main():
    data = [10, 20, 30, 40, 50]
    # An attempt was made to get a slice with a start or end index that was out of range.
    big_slice = data[2:10]  #End index out of bounds 
    assert len(big_slice) == 8  # Expected slice length to be 8, but only 3 elements
    
main()

