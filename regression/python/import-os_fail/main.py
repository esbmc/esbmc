import os
from os import listdir, popen, makedirs, mkdir, rmdir, remove

p = "/tmp/foo/"
f = "/tmp/foo/bar.txt"
q = "/tmp/baz"

popen("ls /tmp")
makedirs(p, exist_ok=True)
listdir(p)

exists = os.path.exists(f)
base = os.path.basename(f)
if exists:
    remove(f)

exists = os.path.exists(q)
if exists:
    rmdir(q)

try:
    rmdir(f)
except OSError:
    assert False
