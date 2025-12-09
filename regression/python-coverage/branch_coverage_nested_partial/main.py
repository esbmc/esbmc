def classify(x: int, y: int) -> int:
    if x > 0:
        if y > 0:
            return 1
        else:
            return 2
    else:
        return 3

# cover some branches
classify(5, 3)
classify(5, -2)
