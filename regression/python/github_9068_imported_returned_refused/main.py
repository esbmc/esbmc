# pick() returns f (str) or g (int), so calling its result is refused (#9068).
from mod import pick

b = pick(True)()
assert b == "a"
