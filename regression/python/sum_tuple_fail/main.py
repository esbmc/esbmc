# sum((1, 2.5, 3)) is 6.5, not 6 (the old truncating behaviour) — the
# assertion must fail.
assert sum((1, 2.5, 3)) == 6
