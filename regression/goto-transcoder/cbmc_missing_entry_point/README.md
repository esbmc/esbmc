A `--function` naming no function in the loaded binary used to reach the
inliner as a dangling call, which aborted (SIGABRT) after logging `failed to
find function`. The binary is a byte copy of `cbmc_class_id_intern.goto`, whose
own fixture pins the accepting side of the same path (`--function main`
verifies); regenerate it with that directory's
`gen_cbmc_class_id_intern.py`.
