import os
from os import listdir

p = "/foo/bar/"

listdir(p)
exists = os.path.exists(p)
base = os.path.basename(p)