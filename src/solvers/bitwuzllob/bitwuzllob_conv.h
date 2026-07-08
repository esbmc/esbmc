#ifndef _ESBMC_SOLVERS_BITWUZLLOB_BITWUZLLOB_CONV_H
#define _ESBMC_SOLVERS_BITWUZLLOB_BITWUZLLOB_CONV_H

#include <solvers/smtlib/smtlib_conv.h>

/** Backend for Bitwuzllob, the integration of the Bitwuzla SMT solver into
 *  the massively parallel Mallob platform (Schreiber, Niemetz, Preiner,
 *  TACAS'26). Mallob is an MPI program that cannot be linked into another
 *  application, so this backend reuses the smtlib backend's SMT-LIB2
 *  serializer to render the formula into a file and invokes Mallob's one-shot
 *  "mono" mode on it (--bitwuzllob-prog, "%f" is replaced by the file path).
 *
 *  A terminated mono process cannot answer (get-value) queries, so when the
 *  formula is satisfiable and a counterexample is required, a local
 *  interactive SMT-LIB2 solver (--bitwuzllob-model-prog) is fed the same
 *  formula through the inherited pipe machinery and serves the model. Without
 *  a model solver, satisfiable results require --result-only. */
class bitwuzllob_convt : public smtlib_convt
{
public:
  bitwuzllob_convt(const namespacet &ns, const optionst &options);
  ~bitwuzllob_convt() override;

  smt_resultt dec_solve() override;
  const std::string solver_text() override;
  std::string dump_smt() override;

private:
  bitwuzllob_convt(
    const namespacet &ns,
    const optionst &options,
    const std::string &formula_path);

  /** Run the Mallob mono-mode command on formula_path and parse the verdict
   *  from its standard output. */
  smt_resultt run_bitwuzllob();

  std::string formula_path;
  bool solved = false;
};

#endif /* _ESBMC_SOLVERS_BITWUZLLOB_BITWUZLLOB_CONV_H */
