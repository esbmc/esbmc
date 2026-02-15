# Test: String removeprefix with absent prefix
text = "sample_text"
result = text.removeprefix("prefix_")
assert result == "sample_text"
assert result != "text"
