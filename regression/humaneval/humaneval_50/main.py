# HumanEval/50
# Entry Point: decode_shift
# ESBMC-compatible format with direct assertions



def encode_shift(s: str):
    """
    returns encoded string by shifting every character by 5 in the alphabet.
    """
    return "".join([chr(((ord(ch) + 5 - ord("a")) % 26) + ord("a")) for ch in s])


def decode_shift(s: str):
    """
    takes as input string encoded with encode_shift function. Returns decoded string.
    """
    return "".join([chr(((ord(ch) - 5 - ord("a")) % 26) + ord("a")) for ch in s])

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert decode_shift(copy.deepcopy(encoded_str)) == str
    print("✅ HumanEval/50 - All assertions completed!")
