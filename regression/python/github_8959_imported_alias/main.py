# An alias another module imports is not dropped (#8959).
from lib import f

assert f(1, 2, 3) == 6
