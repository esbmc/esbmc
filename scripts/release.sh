#!/bin/bash

# Procedure:
# 1) Tell the user what's about to happen
# 2) Produce cleaned tarball via export-pub
# 3) For each target:
# 3a) Extract to an arbitary dir
# 3b) Run configure apropriately
# 3c) make
# 3d) Manufacture release tree
# 3e) Tar up
# 4) Clear up
