import os
from os import listdir, popen, makedirs, remove

p = "/tmp/foo/"
f = "/tmp/foo/bar.txt"

popen("ls /tmp")
makedirs(p, exist_ok=True)
listdir(p)

exists = os.path.exists(f)
base = os.path.basename(f)
if exists:
  remove(f)
