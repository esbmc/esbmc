# bytes.fromhex over an odd-length (or otherwise non-hex) string raises a
# ValueError in CPython. ESBMC must reject it with a clean error rather than
# folding a partial/garbage byte value (a wrong verdict). A space inside a
# byte pair is likewise invalid; only whitespace between pairs is skipped.
x = bytes.fromhex("012")
