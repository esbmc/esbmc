def main():
    lst = [10, 20, 30, 40]
    idx = 10  # Dynamic index, intentionally out of bounds
    value = lst[idx]
    assert value == 0

main()
