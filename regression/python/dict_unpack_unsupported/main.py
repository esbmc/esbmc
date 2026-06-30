# Dict-literal `**` unpacking is not modelled. It must produce a clean
# diagnostic rather than crash the frontend with an uncaught JSON exception
# (the null key emitted for `**m` previously aborted the annotation pass).
m = {"x": 1}
n = {**m, "y": 2}
assert n["y"] == 2
