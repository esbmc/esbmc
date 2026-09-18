#include <transition-system/ts_engines.h>

#include <irep2/irep2_utils.h>
#include <langapi/language_util.h>
#include <solvers/solve.h>
#include <util/base/time_stopping.h>
#include <util/message/message.h>

#include <algorithm>
#include <sstream>

ts_enginet::ts_enginet(
  const transition_systemt &ts,
  const namespacet &ns,
  optionst &options,
  bool incremental)
  : ts(ts), ns(ns)
{
  options.set_option("smt-assumptions", incremental);
  // An incremental Bitwuzla re-runs its preprocessing on every new step with
  // its caches cleared, which costs more than it saves.
  std::vector<std::string> &bitwuzla = options.option_values["bitwuzla-opt"];
  const bool no_preprocess =
    incremental &&
    std::none_of(bitwuzla.begin(), bitwuzla.end(), [](const std::string &o) {
      return o.rfind("preprocess=", 0) == 0;
    });
  if (no_preprocess)
    bitwuzla.push_back("preprocess=0");
  solver.reset(create_solver("", ns, options));
  if (no_preprocess)
    bitwuzla.pop_back();
  if (incremental && !solver->supports_assumptions())
  {
    log_error(
      "The transition-system BMC engine needs a solver with assumptions");
    abort();
  }

  for (const expr2tc &d : ts.prefix_defs)
    solver->define(d);

  // The first state is the initial state outright, which lets the solver
  // fold constant initial values.
  for (const auto &s : ts.states)
    solver->define(equality2tc(ts.at_step(s.pre, 0), s.init));
}

expr2tc ts_enginet::bad_literal(unsigned step) const
{
  return symbol2tc(get_bool_type(), "ts$bad$" + std::to_string(step));
}

void ts_enginet::assume_prefix()
{
  for (const expr2tc &a : ts.prefix_assumes)
    solver->assert_expr(a);
}

bool ts_enginet::check_prefix()
{
  if (ts.prefix_bad.empty())
  {
    assume_prefix();
    return true;
  }
  std::vector<expr2tc> violated;
  for (const auto &p : ts.prefix_bad)
    violated.push_back(p.violated);
  expr2tc lit = symbol2tc(get_bool_type(), "ts$bad$prefix");
  solver->assert_expr(implies2tc(lit, disjunction(violated)));
  if (solver->dec_solve_assuming({lit}) == smt_resultt::P_SATISFIABLE)
  {
    for (const auto &p : ts.prefix_bad)
      if (solver->l_get(p.violated).is_true())
      {
        log_status(
          "\n[Counterexample]\nViolated before the loop:\n  {}\n  {}",
          p.location.as_string(),
          p.comment);
        break;
      }
    return false;
  }
  solver->assert_expr(not2tc(disjunction(violated)));
  assume_prefix();
  return true;
}

smt_resultt ts_enginet::solve_prefix()
{
  std::vector<expr2tc> violated;
  for (const auto &p : ts.prefix_bad)
    violated.push_back(p.violated);
  solver->assert_expr(disjunction(violated));
  return solver->dec_solve();
}

smt_resultt ts_enginet::solve_bound(unsigned k)
{
  assume_prefix();
  for (unsigned j = 0; j < k; j++)
  {
    add_step(j);
    close_step(j);
  }
  add_step(k);
  solver->assert_expr(bad_literal(k));
  fine_timet start = current_time();
  smt_resultt res = solver->dec_solve();
  log_status(
    "TS BMC step {} (fresh solver) solve {}s",
    k,
    time2string(current_time() - start));
  if (res == smt_resultt::P_SATISFIABLE)
    report_counterexample(k);
  return res;
}

void ts_enginet::add_step(unsigned step)
{
  ts_step_renamert at(ts, step);
  for (const expr2tc &d : ts.body_defs)
    solver->define(at(d));
  std::vector<expr2tc> violated;
  for (const auto &p : ts.bad)
    violated.push_back(at(p.violated));
  solver->assert_expr(implies2tc(bad_literal(step), disjunction(violated)));
}

void ts_enginet::close_step(unsigned step)
{
  ts_step_renamert at(ts, step), next(ts, step + 1);
  for (const expr2tc &a : ts.body_assumes)
    solver->assert_expr(at(a));
  solver->assert_expr(at(ts.back_guard));
  for (const auto &p : ts.bad)
    solver->assert_expr(not2tc(at(p.violated)));
  for (size_t i = 0; i < ts.states.size(); i++)
    solver->define(equality2tc(next(ts.states[i].pre), at(ts.states[i].post)));
}

smt_resultt ts_enginet::solve_bad(unsigned step, bool push_pop)
{
  const expr2tc lit = bad_literal(step);
  if (!push_pop)
    return solver->dec_solve_assuming({lit});

  // Convert outside the frame so pop does not discard the literal's term.
  solver->convert_ast(lit);
  solver->push_ctx();
  solver->assert_expr(lit);
  smt_resultt res = solver->dec_solve();
  if (res == smt_resultt::P_SATISFIABLE)
    report_counterexample(step);
  solver->pop_ctx();
  return res;
}

void ts_enginet::report_counterexample(unsigned step)
{
  std::ostringstream out;
  out << "\n[Counterexample]\nTransition-system counterexample at step " << step
      << " (standard BMC: --unwind " << step + 1 << ")\n";
  out << "State 0:";
  for (size_t i = 0; i < ts.states.size(); i++)
    out << " " << ts.states[i].name << " = "
        << from_expr(ns, "", solver->get(ts.at_step(ts.states[i].pre, 0)));
  out << "\n";
  for (unsigned j = 0; j <= step; j++)
  {
    out << "Step " << j << " inputs:";
    for (const auto &[sym, label] : ts.inputs)
      out << " " << label << " = "
          << from_expr(ns, "", solver->get(ts.at_step(sym, j)));
    out << "\n";
  }
  for (const auto &p : ts.bad)
    if (solver->l_get(ts.at_step(p.violated, step)).is_true())
    {
      out << "Violated property:\n  " << p.location.as_string() << "\n  "
          << p.comment << "\n";
      break;
    }
  log_status("{}", out.str());
}

int ts_enginet::bmc(uint64_t max_k, bool push_pop)
{
  if (!check_prefix())
  {
    log_fail("\nVERIFICATION FAILED");
    return 1;
  }
  if (ts.bad.empty())
  {
    log_success("\nVERIFICATION SUCCESSFUL");
    return 0;
  }

  for (uint64_t k = 0; k <= max_k; k++)
  {
    fine_timet start = current_time();
    add_step(k);
    fine_timet added = current_time();
    smt_resultt res = solve_bad(k, push_pop);
    log_status(
      "TS BMC step {}: {} (encode {}s, solve {}s)",
      k,
      res == smt_resultt::P_SATISFIABLE     ? "violated"
      : res == smt_resultt::P_UNSATISFIABLE ? "safe"
                                            : "unknown",
      time2string(added - start),
      time2string(current_time() - added));
    if (res == smt_resultt::P_SATISFIABLE)
    {
      if (!push_pop)
        report_counterexample(k);
      log_fail("\nVERIFICATION FAILED");
      return 1;
    }
    if (res != smt_resultt::P_UNSATISFIABLE)
    {
      log_error("Solver error at step {}", k);
      return 6;
    }
    close_step(k);
  }
  log_fail("\nVERIFICATION UNKNOWN");
  return 0;
}
