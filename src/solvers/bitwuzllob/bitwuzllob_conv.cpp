#include <solvers/bitwuzllob/bitwuzllob_conv.h>
#include <solvers/smtlib/oneshot_process.h>
#include <util/message/message.h>

#include <cstdio>

static std::string prog_command(const optionst &options)
{
  std::string cmd = options.get_option("bitwuzllob-prog");
  return cmd.empty() ? "mallob -mono=%f -mono-app=SMT" : cmd;
}

/* create_new_bitwuzllob_solver now lives in src/solvers/camada. */

bitwuzllob_convt::bitwuzllob_convt(
  const namespacet &ns,
  const optionst &options)
  : bitwuzllob_convt(
      ns,
      options,
      oneshot_process::choose_formula_path(options, "bitwuzllob"))
{
}

bitwuzllob_convt::bitwuzllob_convt(
  const namespacet &ns,
  const optionst &options,
  const std::string &_formula_path)
  : smtlib_convt(
      ns,
      options,
      oneshot_process::model_prog(options, "bitwuzllob"),
      _formula_path),
    formula_path(_formula_path)
{
}

bitwuzllob_convt::~bitwuzllob_convt()
{
  if (oneshot_process::uses_temp_formula(options))
    remove(formula_path.c_str());
}

std::string bitwuzllob_convt::dump_smt()
{
  /* Under --smt-formula-only no solve follows; complete the dump with the
   * (check-sat) like the base class. Under --smt-formula-too our dec_solve()
   * emits the (check-sat) itself: appending one here as well would hand
   * Mallob a formula containing two. The base implementation also reports
   * the destination from the --output option, which this backend redirects
   * to the formula file. */
  if (options.get_bool_option("smt-formula-only"))
    return smtlib_convt::dump_smt();
  log_status("SMT formula written to {}", formula_path);
  /* Formula already written to formula_path; empty return keeps bmc.cpp from
   * overwriting it with this string (issue #6059). */
  return "";
}

smt_resultt bitwuzllob_convt::dec_solve()
{
  if (solved)
  {
    log_error(
      "the bitwuzllob backend supports a single (check-sat) query per run; "
      "incremental strategies are not supported");
    abort();
  }
  solved = true;

  pre_solve();

  /* The (check-sat) goes to both sinks: the formula file for Mallob, and the
   * local model solver's pipe (if configured), which starts solving in
   * parallel and only gets waited for when a model is actually needed. The
   * model solver only produces counterexamples, so if it has died (e.g. it
   * failed to start), disable it and let Mallob decide: an unsat proof needs
   * no model, and a sat result reports the missing-model error below rather
   * than crashing on an uncaught exception. */
  try
  {
    emit_check_sat();
  }
  catch (const external_process_died &)
  {
    log_warning(
      "bitwuzllob: the local model solver '{}' terminated unexpectedly (e.g. "
      "it failed to start); continuing without counterexample support",
      options.get_option("bitwuzllob-model-prog"));
    emit_proc.terminate();
    flush(); // complete the formula file for Mallob now that the pipe is gone
  }

  smt_resultt res = oneshot_process::run_solver(
    prog_command(options), formula_path, "bitwuzllob");
  if (res != P_SATISFIABLE)
  {
    /* No model will be read; stop the local solver we fed in parallel rather
     * than let it keep solving until this object is destroyed. */
    emit_proc.terminate();
    return res;
  }

  /* A satisfiable formula needs a live model solver to turn into a
   * counterexample. It is absent either because the model solver died above
   * (a command was given) or was never configured; the message distinguishes
   * the two, but neither is a reason to crash. */
  if (!emit_proc)
  {
    if (options.get_bool_option("result-only"))
      return P_SATISFIABLE;
    if (options.get_option("bitwuzllob-model-prog").empty())
      log_error(
        "bitwuzllob: formula is satisfiable, but building the counterexample "
        "requires a local interactive SMT-LIB2 solver; re-run with "
        "--bitwuzllob-model-prog <cmd> (e.g. \"z3 -in\") or with "
        "--result-only");
    else
      log_error(
        "bitwuzllob: the local model solver is unavailable; cannot build a "
        "counterexample");
    return P_ERROR;
  }

  smt_resultt model_res;
  try
  {
    model_res = read_check_sat_response();
  }
  catch (const external_process_died &)
  {
    log_error(
      "bitwuzllob: the local model solver is unavailable; cannot build a "
      "counterexample");
    return P_ERROR;
  }
  if (model_res != P_SATISFIABLE)
  {
    log_error(
      "bitwuzllob: Bitwuzllob reported sat but the local model solver did "
      "not; refusing to build a counterexample from a diverging model");
    abort();
  }
  return P_SATISFIABLE;
}

const std::string bitwuzllob_convt::solver_text()
{
  return "Bitwuzllob '" + prog_command(options) + "'";
}
