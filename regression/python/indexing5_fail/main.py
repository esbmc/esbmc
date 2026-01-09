def main():
    matrix = [
        [1, 2, 3],
        [4, 5, 6]
    ]
    value = matrix[2][0]  # Access the third row (index=2), there are actually only two rows
    assert value == 0

main()
