# Adapted from TypeEvalPy (https://github.com/secure-software-engineering/TypeEvalPy
# @ 3719de1), micro-benchmark/python_features/assignments/walrus/main.py
#
# The input string is chosen nondeterministically between the original one
# and the empty string, and the word count is used as a divisor.

# Function contains usage of walrus operator

def count_words(string):
    words = string.split()
    word_count = 0
    while words and (word := words.pop()):
        print(word)
        word_count += 1
    return word_count


def main() -> None:
    if nondet_bool():
        s = "Hello Python"
    else:
        s = ""
    a = count_words(s)
    avg = len(s) / a


main()
