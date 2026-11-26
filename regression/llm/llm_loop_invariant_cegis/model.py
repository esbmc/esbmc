#!/usr/bin/env python3
"""Scripted stand-in for a language model: one answer per round."""
import sys

prompt = sys.stdin.read()
if prompt.count("\nuser: ") == 1:
    print("x and y count the iterations.\n<invariants>\n"
          "main L1: x + y == i\nmain L1: x == y\nmain L1: foo(i)\n</invariants>")
else:
    print("i never passes n.\n<invariants>\nmain L1: i <= n\n</invariants>")
