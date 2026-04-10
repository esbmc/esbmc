# Test: String min e max
chars = "abcde"
assert min(chars) == "a"
assert max(chars) == "e"
mixed = "aZbY"
assert min(mixed) == "Y"  # uppercase antes no ASCII
assert max(mixed) == "b"  # lowercase maior
