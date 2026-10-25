# A star import rebinding a class imported by name keeps its nested class
# unrewritten (#9047).
from mod import A
from other import *

assert A.B != 5
