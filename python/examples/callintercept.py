#!/usr/bin/python3
import sys
import esbmc
from esbmc.goto_programs import goto_program_instruction_type as gptypes

# Arguments: Anything on the command line, and callintercept.c

args = sys.argv[1:] # This may be an empty list
args.append('callintercept.c')
ns, opts, po = esbmc.init_esbmc_process(args)

class ExState2(esbmc.symex.execution_state.dfs_execution_state):
    def __init__(self, *posargs):
        # Call superclass constructor
        super(ExState2, self).__init__(*posargs)

    def symex_step(self, art):
        # Pick out src obj
        gss = art.get_cur_state().get_active_state()
        src = gss.source
        # pc becomes an insn number, lookup insns. Select current insn
        insns = src.prog.get_instructions()
        localpc = src.pc - insns[0].location_number
        insn = src.prog.get_instructions()[localpc]

        normal = True
        if insn.type == gptypes.FUNCTION_CALL:
            # Pick out function calls
            call = esbmc.downcast_expr(insn.code)
            sym = esbmc.downcast_expr(call.function)

            # Pick out the desired function
            if sym.name.as_string() == 'c::foobar':
                # Build a constant for 'one'
                bigint = esbmc.BigInt(1)
                ubv = esbmc.type.unsignedbv.make(32)
                one = esbmc.expr.constant_int.make(ubv, bigint)

                # Assign one to the given return symbol
                exobj.symex_assign_symbol(call.return_sym, one, gss.guard)

                # Increment the program counter _past_ the call we just
                # interpreted, set normal to indicate this shouldn't be
                # interpreted by the usual symex_step operation.
                src.pc += 1
                normal = False
        
        if normal: 
            super(ExState2, self).symex_step(art)

eq = esbmc.symex.equation(ns)
art = esbmc.symex.reachability_tree(po.goto_functions, ns, opts, eq, po.context, po.message_handler)

exobj = ExState2(po.goto_functions, ns, art, eq, po.context, opts, po.message_handler)

art.setup_for_new_explore()
art.execution_states = [exobj]
art.set_cur_state(exobj)

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
