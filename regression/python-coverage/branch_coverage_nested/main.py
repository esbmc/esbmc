def classify(x: int, y: int) -> int:
    if x > 0:
        if y > 0:
            return 1
        else:
            return 2
    else:
        if y > 0:
            return 3
        else:
            return 4

# Cover all branches
classify(5, 3)
classify(5, -2)
classify(-1, 3)
classify(-1, -2)
