# github #9960: a tuple element keeps the element-wise membership check.
assert [1] not in [(1, )]
assert [1] in [(1, )]
