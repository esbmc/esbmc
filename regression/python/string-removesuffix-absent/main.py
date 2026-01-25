# Test: String removesuffix with absent suffix
text = "sample_text"
result = text.removesuffix("_suffix")
assert result == "sample_text"
assert result != "sample"
