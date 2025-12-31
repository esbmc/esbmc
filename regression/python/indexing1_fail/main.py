def main():
    lst = [10, 20, 30]
    value = lst[5]  # <-- Deliberate out-of-bounds access
    assert value == 0  # Add an assert to make model checking easier to expose problems

main()
