# #10223: a from-import of a name keeps the raising module unconverted.
from broken import g

assert g() == 3
