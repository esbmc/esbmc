# A loop bounded by len() must cost the same at any --unwind: the bound folds
# to the list's length. Before len() returned the size directly this was 1270,
# 2230 and 3190 symex assignments at --unwind 20, 40 and 60; it is now 418.
# In VCCs at --unwind 40: 1380 before, 276 after.
xs = [1, 2, 3, 4]
i: int = 0
total: int = 0
while i < len(xs):
    total = total + xs[i]
    i = i + 1
assert total == 10
