# Sorting an all-integer list must not cost anything that scales with --unwind:
# the numeric arms are selected by the frontend's literal type_flag, so the
# lexicographic memcmp arm is never symbolically executed here. Before that
# dispatch was reordered this generated 51088 VCCs at --unwind 32.
xs = [7, 6, 5, 4, 3, 2, 1, 0]
xs.sort()
assert xs == [0, 1, 2, 3, 4, 5, 6, 7]
