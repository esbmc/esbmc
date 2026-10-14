# min()/max() over a single mixed int/float tuple must keep the wider
# (float) candidate, not truncate it to the current component's int type.
# Regresses the bug where the running double accumulator was cast back to
# int by the select's result type, so min((3, 2.5, 4)) wrongly gave 2.
assert min((3, 2.5, 4)) == 2.5
assert max((1, 5.5, 2)) == 5.5
assert min((10, 2.5, 7, 1)) == 1
assert max((3, 2.5)) == 3
