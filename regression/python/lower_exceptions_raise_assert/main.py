# The "assert it raised" idiom: an unconditional raise inside a try makes the
# normal-completion path dead, so remove_unreachable prunes the try's empty
# CATCH pop, leaving the region unbalanced. The lowering rebalances the unclosed
# region before recovering it (#5075). Here ValueError is caught, so SUCCESSFUL.
try:
    raise ValueError("boom")
    assert False  # unreachable: the raise always fires
except ValueError:
    pass
