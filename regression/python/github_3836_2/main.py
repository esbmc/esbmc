# Unannotated int param in ordering comparison with subtraction (simpler case)
def count_down(n):
    if n <= 0:
        return 0
    return count_down(n - 1) + 1


assert count_down(5) == 5
