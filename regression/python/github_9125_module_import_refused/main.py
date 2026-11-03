# The same as github_9125_from_import_refused, with mod importing mod2 by name.
from mod import pick

b = pick(True)()
assert b == "a"
