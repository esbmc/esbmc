# JSON carries a complex literal as the string "0j", which would box as a
# non-empty - hence truthy - str. Refused rather than answered wrongly.
x = 0j
assert not x
