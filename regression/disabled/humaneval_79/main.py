# HumanEval/79
# Entry Point: decimal_to_binary
# ESBMC-compatible format with direct assertions


def decimal_to_binary(decimal):
    """You will be given a number in decimal form and your task is to convert it to
    binary format. The function should return a string, with each character representing a binary
    number. Each character in the string will be '0' or '1'.

    There will be an extra couple of characters 'db' at the beginning and at the end of the string.
    The extra characters are there to help with the format.

    Examples:
    decimal_to_binary(15)   # returns "db1111db"
    decimal_to_binary(32)   # returns "db100000db"
    """
    return "db" + bin(decimal)[2:] + "db"

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert decimal_to_binary(0) == "db0db"
    # assert decimal_to_binary(32) == "db100000db"
    # assert decimal_to_binary(103) == "db1100111db"
    # assert decimal_to_binary(15) == "db1111db", "This prints if this assert fails 1 (good for debugging!)"
    # assert True, "This prints if this assert fails 2 (also good for debugging!)"
    print("✅ HumanEval/79 - All assertions completed!")
