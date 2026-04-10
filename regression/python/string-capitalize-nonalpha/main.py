# Test: String capitalize with non-alphabetic first character
non_alpha = "1sample text"
non_alpha_cap = non_alpha.capitalize()
assert non_alpha_cap == "1sample text"
assert non_alpha_cap != "1Sample text"
