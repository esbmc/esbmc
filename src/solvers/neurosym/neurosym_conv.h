#ifndef _ESBMC_SOLVERS_NEUROSYM_NEUROSYM_CONV_H
#define _ESBMC_SOLVERS_NEUROSYM_NEUROSYM_CONV_H

#include <solvers/smtlib/smtlib_conv.h>

/** Backend for NeuroSym, a neural-guided SMT solver (a GAN proposes candidate
 *  models, with a Z3 fallback preserving soundness and completeness). NeuroSym
 *  natively parses only the QF_BV and QF_LIA fragments of SMT-LIB2, and this
 *  backend drives it as a pure QF_BV solver. NeuroSym is a Python program that
 *  cannot be linked into another application, so this backend reuses the
 *  smtlib backend's SMT-LIB2 serializer to render the formula into a file and
 *  runs NeuroSym on it in one-shot batch mode (--neurosym-prog, "%f" is
 *  replaced by the file path).
 *
 *  NeuroSym has no native array, floating-point, or tuple support, so the
 *  factory leaves those capability interfaces unset and ESBMC's flatteners
 *  lower everything to pure QF_BV; the header emitted before the formula is
 *  overridden to (set-logic QF_BV) accordingly. Integer/real encoding (--ir)
 *  is rejected in solve.cpp because the flattened int-mode logic would be
 *  QF_AUFLIRA, which NeuroSym cannot parse.
 *
 *  A terminated batch process cannot answer (get-value) queries, so when the
 *  formula is satisfiable and a counterexample is required, a local
 *  interactive SMT-LIB2 solver (--neurosym-model-prog) is fed the same
 *  formula through the inherited pipe machinery and serves the model. Without
 *  a model solver, satisfiable results require --result-only. */
class neurosym_convt : public smtlib_convt
{
public:
  neurosym_convt(const namespacet &ns, const optionst &options);
  ~neurosym_convt() override;

  smt_resultt dec_solve() override;
  const std::string solver_text() override;
  std::string dump_smt() override;

private:
  neurosym_convt(
    const namespacet &ns,
    const optionst &options,
    const std::string &formula_path);

  std::string formula_path;
  bool solved = false;
};

#endif /* _ESBMC_SOLVERS_NEUROSYM_NEUROSYM_CONV_H */
