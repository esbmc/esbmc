#ifndef _ESBMC_SOLVERS_CAMADA_ONESHOT_OPTIONS_H
#define _ESBMC_SOLVERS_CAMADA_ONESHOT_OPTIONS_H

#include <string>

class optionst;

/** Option and temporary-file policy for the backends that render the formula
 *  to a file and hand it to an external one-shot solver (bitwuzllob,
 *  neurosym). The spawn, the verdict scan and the parallel model solver live
 *  in camada's SMTLIBSolver one-shot mode; what stays here is the part that
 *  depends on ESBMC's options and its signal-safe temp-file registry.
 *  `name` is the backend's option prefix: options are looked up as
 *  <name>-model-prog etc. and warnings are prefixed with it. */
namespace oneshot_options
{
/** Whether the formula file is a temporary the backend creates, as opposed
 *  to a user-supplied --output path (or stdout under --smt-formula-only). */
bool uses_temp_formula(const optionst &options);

/** The formula file the external solver is pointed at. Honour --output so the
 *  user can keep the formula; otherwise use a self-cleaning temporary. With
 *  --smt-formula-only no solver runs, so honour --output including stdout via
 *  "-" (the default when no file is given). */
std::string choose_formula_path(const optionst &options, const char *name);

/** The local interactive model solver command (<name>-model-prog), or "" when
 *  none is configured. With --result-only no counterexample is ever built
 *  (bmc.cpp skips trace construction), so feeding the formula to a local
 *  model solver would only start a solve whose answer is never read; the
 *  option is ignored with a warning in that case. */
std::string model_prog(const optionst &options, const char *name);
} // namespace oneshot_options

#endif /* _ESBMC_SOLVERS_CAMADA_ONESHOT_OPTIONS_H */
