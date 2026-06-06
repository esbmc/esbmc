# Non-vacuity counterpart of github_5163_join_appended_list: joining an
# append-built list returns the genuine value ("is"), so asserting the old
# buggy result ("") must FAIL with a counterexample -- proving the runtime
# join reflects the real list contents rather than the [] initializer (#5163).

new_lst = []
new_lst.append("is")
assert " ".join(new_lst) == ""
