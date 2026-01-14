# Test: Método strip - deve PASSAR
text = "  hello  "
stripped = text.strip()
assert stripped == "hello"
assert len(stripped) == 5
