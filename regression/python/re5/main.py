import re
assert re.match("a.*", "abc")
assert not re.match("b.*", "abc")
