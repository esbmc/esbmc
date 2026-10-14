# Exercises Python exception handling under --irep2-bodies (V.4.3, esbmc#4715).
# abs() on a str raises TypeError; the try/except catches it, so verification
# must SUCCEED. This pins both the cpp-throw round-trip (the raise must not be
# dropped) and the cpp-catch round-trip (the try/handler operands must survive
# migrate_expr -> migrate_expr_back).
try:
    x = abs("hello")
except TypeError:
    pass
