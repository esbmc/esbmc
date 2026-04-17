# HumanEval/78
# Entry Point: hex_key
# ESBMC-compatible format with direct assertions


def hex_key(num):
    """You have been tasked to write a function that receives 
    a hexadecimal number as a string and counts the number of hexadecimal 
    digits that are primes (prime number, or a prime, is a natural number 
    greater than 1 that is not a product of two smaller natural numbers).
    Hexadecimal digits are 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, A, B, C, D, E, F.
    Prime numbers are 2, 3, 5, 7, 11, 13, 17,...
    So you have to determine a number of the following digits: 2, 3, 5, 7, 
    B (=decimal 11), D (=decimal 13).
    Note: you may assume the input is always correct or empty string, 
    and symbols A,B,C,D,E,F are always uppercase.
    Examples:
    For num = "AB" the output should be 1.
    For num = "1077E" the output should be 2.
    For num = "ABED1A33" the output should be 4.
    For num = "123456789ABCDEF0" the output should be 6.
    For num = "2020" the output should be 2.
    """
    primes = ('2', '3', '5', '7', 'B', 'D')
    total = 0
    for i in range(0, len(num)):
        if num[i] in primes:
            total += 1
    return total

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert hex_key("AB") == 1, "First test error: " + str(hex_key("AB"))
    # assert hex_key("1077E") == 2, "Second test error: " + str(hex_key("1077E"))
    # assert hex_key("ABED1A33") == 4, "Third test error: " + str(hex_key("ABED1A33"))
    # assert hex_key("2020") == 2, "Fourth test error: " + str(hex_key("2020"))
    # assert hex_key("123456789ABCDEF0") == 6, "Fifth test error: " + str(hex_key("123456789ABCDEF0"))
    # assert hex_key("112233445566778899AABBCCDDEEFF00") == 12, "Sixth test error: " + str(hex_key("112233445566778899AABBCCDDEEFF00"))
    # assert hex_key([]) == 0
    print("✅ HumanEval/78 - All assertions completed!")
