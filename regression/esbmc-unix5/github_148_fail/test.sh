#!/bin/bash
esbmc main.c >esbmc_out.txt 2>&1
grep 'Fetching array elements inside tuples currently unimplemented, sorry' esbmc_out.txt >/dev/null 2>&1
