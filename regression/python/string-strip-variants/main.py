# Test: String lstrip e rstrip
text = "  hello  "
left = text.lstrip()
assert left == "hello  "
right = text.rstrip()
assert right == "  hello"
both = text.strip()
assert both == "hello"
