# Test: String title with numbers
text = "item 2 title"
text_title = text.title()
assert text_title == "Item 2 Title"
assert text_title != "Item 2 title"
