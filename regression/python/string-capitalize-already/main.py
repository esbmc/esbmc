# Test: String capitalize on already capitalized string
already = "Sample Text"
already_cap = already.capitalize()
assert already_cap == "Sample text"
assert already_cap != "Sample Text"
