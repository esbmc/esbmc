# Companion to github_4782_object_size: the array success path of the
# __ESBMC_get_object_size intrinsic must keep verifying. len() on a string
# lowers to __ESBMC_get_object_size on a char array, which the hardened
# intrinsic handles exactly as before.
s = "hello"
assert len(s) == 5
