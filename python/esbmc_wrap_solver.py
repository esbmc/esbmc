#!/usr/bin/python3
import sys
import esbmc

# Cruft for running below. Pass command line options through to esbmc.
# Hopefully this means we effectively wrap esbmc.

ns, opts, po = esbmc.init_esbmc_process(sys.argv[1:])
eq = esbmc.symex.equation(ns)
art = esbmc.symex.reachability_tree(po.goto_functions, ns, opts, eq, po.context, po.message_handler)

art.setup_for_new_explore()
result = art.get_next_formula()
if result.remaining_claims == 0:
    print('No remaining claims')
    print("VERIFICATION SUCCESSFUL")
    sys.exit(0)

esbmc.symex.slice(result.target)
lolsolve = esbmc.solve.solvers.z3.make(False, ns, opts)
result.target.convert(lolsolve)
issat = lolsolve.dec_solve()

if issat == esbmc.solve.smt_result.sat:
    trace = esbmc.symex.goto_tracet()
    esbmc.symex.build_goto_trace(result.target, lolsolve, trace)
    print(trace.to_string(ns))
    print("VERIFICATION FAILED")
elif issat == esbmc.solve.smt_result.unsat:
    print("VERIFICATION SUCCESSFUL")
else:
    print("haha error")
