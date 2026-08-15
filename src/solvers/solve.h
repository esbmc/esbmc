#ifndef _ESBMC_SOLVERS_SOLVE_H_
#define _ESBMC_SOLVERS_SOLVE_H_

#include <memory>
#include <camada/camada.h>
#include <util/config/config.h>
#include <util/symtab/namespace.h>
#include <util/message/message.h>

class smt_convt;
class smt_solver_baset;

/** Builds the camada solver for one backend, configured from @p options.
 *
 *  The linked backends differ only in this: which camada solver to construct.
 *  So the selection table holds these directly and one shared factory
 *  (create_linked_solver) wraps whichever it picks -- there is no per-backend
 *  ESBMC factory, and no backend tag to switch on.
 */
using camada_buildert = camada::SMTSolverRef (*)(const optionst &options);

camada::SMTSolverRef create_esbmc_z3_solver(const optionst &options);
camada::SMTSolverRef create_esbmc_cvc5_solver(const optionst &options);
camada::SMTSolverRef create_esbmc_mathsat_solver(const optionst &options);
camada::SMTSolverRef create_esbmc_yices_solver(const optionst &options);
camada::SMTSolverRef create_esbmc_bitwuzla_solver(const optionst &options);

/** Wraps a linked camada solver in the ESBMC solver object. */
std::unique_ptr<smt_solver_baset> create_linked_solver(
  const optionst &options,
  const namespacet &ns,
  camada_buildert build);

/** The SMT-LIB backend is the exception: it chooses between a one-shot external
 *  program and an interactive/write-only script, and rejects strategies the
 *  one-shot shape cannot serve. */
std::unique_ptr<smt_solver_baset>
create_new_smtlib_solver(const optionst &options, const namespacet &ns);

std::unique_ptr<smt_convt>
create_solver(const namespacet &ns, const optionst &options);

/// Abort early if the user explicitly selected an SMT solver that was not
/// built into this ESBMC binary. Safe to call before parsing the program;
/// returns silently when no explicit selection was made (the default picker
/// will choose from whatever is built in at solver-creation time).
void check_solver_availability(const optionst &options);

#endif
