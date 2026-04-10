# Test: String format_map missing key (expected to fail)
text = "{x}"
text.format_map({"y": 1})
