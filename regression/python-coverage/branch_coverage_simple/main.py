def check_sign(n: int) -> int:
    if n > 0:
        return 1
    else:
        return -1

# Cover both branches
check_sign(5)
check_sign(-3)
