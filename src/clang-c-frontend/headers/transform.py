#!/usr/bin/env python

import os
import argparse

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('path', help='Path containing the headers', default='./', type=str)
  args = parser.parse_args()

  p = args.path
  files = [f for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))]
  files = [f for f in files if f.endswith(".h")]
  files.sort()

  print('extern "C"\n{\n')
  for filename in files:
    name, ext = os.path.splitext(filename)
    assert(len(ext) == 2)

    name_ = name.replace('-', '_')
    print("extern char " + name_ + "_buf[];")
    print("extern unsigned int " + name_ + "_buf_size;\n")

  print('struct hooked_header clang_headers[] = {')
  for filename in files:
    name, ext = os.path.splitext(filename)
    assert(len(ext) == 2)

    name_ = name.replace('-', '_')
    print(f'{{"{filename}", {name_}_buf, &{name_}_buf_size}},')

    os.rename(filename, name_ + ext)

  print('{nullptr, nullptr, nullptr}};\n}')

  for filename in files:
    print(filename.replace('-', '_'), end=' ')

