# Test: String title on already titled string
already = "Sample Text"
already_title = already.title()
assert already_title == "Sample Text"
assert already_title != "Sample text"
