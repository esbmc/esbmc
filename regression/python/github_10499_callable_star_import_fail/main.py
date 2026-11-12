# github #10499: a star import may rebind a builtin name
from m import *

assert callable(len)
