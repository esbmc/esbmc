def main():
    lst = [10, 20, 30]
    value = lst[-5]  # <-- Deliberate negative out-of-bounds access
    assert value == 0  # Add an assert to make it easier for the model detector to expose problems

main()
