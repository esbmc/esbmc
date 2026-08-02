#ifndef _ESBMC_SOLVERS_SMTLIB_ONESHOT_PROCESS_H
#define _ESBMC_SOLVERS_SMTLIB_ONESHOT_PROCESS_H

#include <solvers/smt/smt_result.h>

#include <optional>
#include <string>

class optionst;

/** Helpers shared by backends that render the formula into a file via the
 *  smtlib serializer and run an external one-shot solver command on it
 *  (bitwuzllob, neurosym). `name` is the backend's option prefix: options are
 *  looked up as <name>-model-prog etc. and log messages are prefixed with
 *  it. */
namespace oneshot_process
{
/** Recognize a verdict on one output line, tolerating surrounding whitespace.
 *  Accepts the bare SMT-LIB verdicts as well as SAT-competition-style "s ..."
 *  lines, amid arbitrary log noise. `unknown` maps to P_ERROR. */
std::optional<smt_resultt> parse_verdict_line(const std::string &line);

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

/** Run the one-shot solver command on formula_path and parse the verdict from
 *  its standard output. Every %f in cmd is replaced by the (shell-quoted)
 *  formula path; if no %f is present, the path is appended. */
smt_resultt run_solver(
  const std::string &cmd,
  const std::string &formula_path,
  const char *name);
} // namespace oneshot_process

#endif /* _ESBMC_SOLVERS_SMTLIB_ONESHOT_PROCESS_H */
