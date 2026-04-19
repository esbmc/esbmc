text: str = "hi"
assert text[-3] == "h"  # invalid: index -3 < -len("hi")
