import os
from os import mkdir, rmdir, remove

try:
    mkdir("/tmp/mydir")
except FileExistsError:
    print("Directory already exists")

try:
    rmdir("/tmp/mydir")
except OSError:
    print("Directory not empty or other OS error")

try:
    remove("/tmp/myfile.txt")
except FileNotFoundError:
    print("File not found")
