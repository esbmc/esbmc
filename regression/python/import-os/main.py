import os
from os import listdir, popen, makedirs, remove

p = "/tmp/foo/"
f = "/tmp/foo/bar.txt"

popen(p)
makedirs(p)
listdir(p)

exists = os.path.exists(p)
base = os.path.basename(f)
remove(f)