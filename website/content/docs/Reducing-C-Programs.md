---
title: Reducing C Programs
---


When reporting an issue you should always try to use [C-Reduce](https://embed.cs.utah.edu/creduce/). In Ubuntu, this is available on apt: `apt install creduce`


# Structure

The usual structure for you to run c-reduce is:

- `main.i` (file containing the issue)
- `test.sh` (script that will be executed by c-reduce)
- `esbmc` (binary that contains the issue)

# Generating a preprocessed file

To generate a `.i` from a `.c` file you should use the same compiler used by the language frontend i.e Clang 11

`clang -E main.c > main.i`

# Creating the script for c-reduce

You have to create a script that makes an exit core 0 if the failure that you want has happened. Examples at the Cookbook section

# Running

1. Ensure that `test.sh` have execution permission: `chmod +x test.sh`
1. Check if `esbmc` in on the path: `esbmc --help`
1. Based on your specs you should define the number of threads for creduce. In general 9 is a good value

Then run:

`creduce --n 9 test.sh main.i`

The original file will be saved as `main.i.orig` and after finishing the file `main.i` will contain the reduced program.

# Cookbook

Here are some scripts for common issues:

## Segmentation Fault, Abort and similar issues

You can get the result value of esbmc (`$?`) and wrap it into an `if` statement

```bash
#!/bin/bash
esbmc main.i
if [ $? -eq 139 ] # Segmentation fault
then
    exit 0
fi

exit 1
```

## When ESBMC throws a message but exits normally

This example would check if the output contains the statement `Unexpected side-effect statement`.
It is using the `goto-functions-only` parameter which can be replaced.

```bash
#!/bin/bash
esbmc main.i --goto-functions-only >esbmc_out.txt 2>&1
grep 'Unexpected side-effect statement' esbmc_out.txt >/dev/null 2>&1
```

## When ESBMC fails but other tools find the result correctly

In this example ESBMC output contains `VERIFICATION SUCCESFUL` and the other tool `Verification result: FALSE`. You can replace this at will

```bash
#!/bin/bash

esbmc race-1_1-join.c --incremental-bmc --context-bound 2 --no-pointer-check --no-bounds-check  --no-div-by-zero-check --no-slice --yices --no-por >esbmc_out.txt 2>&1
cbmc race-1_1-join.c >cbmc_out.txt 2>&1
grep 'VERIFICATION FAILED' esbmc_out.txt >/dev/null 2>&1 &&\
grep 'VERIFICATION SUCCESSFUL' cbmc_out.txt >/dev/null 2>&1
```