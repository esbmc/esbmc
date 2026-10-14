# Pins Mode-C soundness gate G2 for Python: when a reachability proof is
# truncated by --unwind below the iteration at which __ESBMC_unreachable() is
# reachable, ESBMC must surface the truncation via an unwinding assertion
# rather than silently report SUCCESSFUL. The site below is reachable only
# at i == 9 (the 10th iteration); --unwind 5 truncates the loop before that
# point, so the unwinding assertion must fire.
#
# Pairing with --no-unwinding-assertions would silently turn this case
# SUCCESSFUL (false positive on a provably-reachable site); that flag is
# therefore banned for Python reachability proofs.

i: int = 0
N: int = 10
while i < N:
    if i == 9:
        __ESBMC_unreachable()
    i = i + 1
