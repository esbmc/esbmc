
def levenshtein(source, target):
    if source == '' or target == '':
        return len(source) or len(target)

    elif source[0] == target[0]:
        return levenshtein(source[1:], target[1:])

    else:
        return 1 + min(
            levenshtein(source,     target[1:]),
            levenshtein(source[1:], target[1:]),
            levenshtein(source[1:], target)
        )


assert levenshtein("electron", "neutron") == 3
assert levenshtein("kitten", "sitting") == 3
assert levenshtein("rosettacode", "raisethysword") == 8
# assert levenshtein(
#     "amanaplanacanalpanama",
#     "docnoteidissentafastneverpreventsafatnessidietoncod"
# ) == 42
assert levenshtein("abcdefg", "gabcdef") == 2
assert levenshtein("", "") == 0
assert levenshtein("hello", "olleh") == 4
