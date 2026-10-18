# A declared type governs the whole scope, so the later assignment is the
# defect - not the annotated statement, which is usually right.
x: int = 5
x = "now a str"
assert x == "now a str"
