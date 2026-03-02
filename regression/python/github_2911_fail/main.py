only_spaces: bool = False

title = "   "
for char in title:
    if char.isspace():
        only_spaces = False

assert only_spaces
