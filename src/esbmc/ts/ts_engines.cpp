#include <esbmc/ts/ts_engines.h>

#include <irep2/irep2_utils.h>
#include <langapi/language_util.h>
#include <solvers/solve.h>
#include <util/base/time_stopping.h>
#include <util/message/message.h>

#include <algorithm>
#include <map>
#include <sstream>

ts_enginet::ts_enginet(
  const transition_systemt &ts,
  const namespacet &ns,
  optionst &options,
  bool bind_init,
  bool incremental,
  bool cores)
  : ts(ts), ns(ns)
{
  options.set_option("smt-assumptions", incremental);
  options.set_option("smt-unsat-assumptions", incremental && cores);
  // Without cores, an incremental Bitwuzla re-runs its preprocessing on every
  // new step with its caches cleared, which costs more than it saves.
  std::vector<std::string> &bitwuzla = options.option_values["bitwuzla-opt"];
  const bool no_preprocess =
    incremental && !cores &&
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
    log_error("The transition-system engines need a solver with assumptions");
    abort();
  }
  act_init = symbol2tc(get_bool_type(), "ts$act$init");
  act_inv = symbol2tc(get_bool_type(), "ts$act$inv");

  for (const expr2tc &d : ts.prefix_defs)
    solver->define(d);
  for (const expr2tc &a : ts.prefix_assumes)
    solver->assert_expr(a);

  std::vector<expr2tc> init;
  for (size_t i = 0; i < ts.state_pre.size(); i++)
  {
    expr2tc eq = equality2tc(ts.at_step(ts.state_pre[i], 0), ts.state_init[i]);
    if (bind_init)
      solver->define(eq);
    init.push_back(eq);
  }
  solver->assert_expr(implies2tc(act_init, conjunction(init)));
}

expr2tc ts_enginet::bad_literal(unsigned step) const
{
  return symbol2tc(get_bool_type(), "ts$bad$" + std::to_string(step));
}

bool ts_enginet::check_prefix()
{
  if (ts.prefix_bad.empty())
    return true;
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
  for (const expr2tc &inv : ts.invariants)
    solver->assert_expr(implies2tc(act_inv, at(inv)));
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
  for (size_t i = 0; i < ts.state_pre.size(); i++)
    solver->define(
      equality2tc(next(ts.state_pre[i]), at(ts.state_post[i])));
}

smt_resultt ts_enginet::solve_bad(unsigned step, bool from_init, bool push_pop)
{
  std::vector<expr2tc> lits{bad_literal(step)};
  lits.push_back(from_init ? act_init : act_inv);
  if (!push_pop)
    return solver->dec_solve_assuming(lits);

  // Convert outside the frame so pop does not discard the literals' terms.
  for (const expr2tc &l : lits)
    solver->convert_ast(l);
  solver->push_ctx();
  for (const expr2tc &l : lits)
    solver->assert_expr(l);
  smt_resultt res = solver->dec_solve();
  if (res == smt_resultt::P_SATISFIABLE)
    report_counterexample(step);
  solver->pop_ctx();
  return res;
}

void ts_enginet::report_counterexample(unsigned step)
{
  std::ostringstream out;
  out << "\n[Counterexample]\nTransition-system counterexample at step "
      << step << " (standard BMC: --unwind " << step + 1 << ")\n";
  out << "State 0:";
  for (size_t i = 0; i < ts.state_pre.size(); i++)
    out << " " << ts.state_names[i] << " = "
        << from_expr(ns, "", solver->get(ts.at_step(ts.state_pre[i], 0)));
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
    smt_resultt res = solve_bad(k, true, push_pop);
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

unsigned ts_enginet::add_simple_path_lemmas(unsigned step)
{
  if (ts.state_pre.empty())
    return 0;
  // Read the whole model before asserting: asserting discards it.
  std::map<std::vector<expr2tc>, unsigned> seen;
  std::vector<std::pair<unsigned, unsigned>> repeats;
  for (unsigned i = 0; i <= step; i++)
  {
    std::vector<expr2tc> values;
    for (const expr2tc &pre : ts.state_pre)
      values.push_back(solver->get(ts.at_step(pre, i)));
    auto [it, fresh] = seen.emplace(values, i);
    if (!fresh)
      repeats.emplace_back(it->second, i);
  }
  for (const auto &[i, j] : repeats)
  {
    std::vector<expr2tc> differ;
    for (const expr2tc &pre : ts.state_pre)
      differ.push_back(notequal2tc(ts.at_step(pre, i), ts.at_step(pre, j)));
    solver->assert_expr(disjunction(differ));
  }
  return repeats.size();
}

int ts_enginet::k_induction(uint64_t max_k, bool simple_path)
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

    smt_resultt base = solve_bad(k, true, false);
    if (base == smt_resultt::P_SATISFIABLE)
    {
      report_counterexample(k);
      log_fail("\nVERIFICATION FAILED");
      return 1;
    }
    if (base != smt_resultt::P_UNSATISFIABLE)
    {
      log_error("Solver error in the base case at step {}", k);
      return 6;
    }
    fine_timet base_done = current_time();

    unsigned lemmas = 0;
    smt_resultt step_res;
    while (true)
    {
      step_res = solve_bad(k, false, false);
      if (step_res != smt_resultt::P_SATISFIABLE || !simple_path)
        break;
      unsigned added = add_simple_path_lemmas(k);
      if (added == 0)
        break;
      lemmas += added;
    }
    log_status(
      "TS k-induction k = {}: base {}s, step {}s, {} simple-path lemmas",
      k,
      time2string(base_done - start),
      time2string(current_time() - base_done),
      lemmas);

    if (step_res == smt_resultt::P_UNSATISFIABLE)
    {
      log_success(
        "\nSolution found by the transition-system inductive step (k = {})"
        "\nVERIFICATION SUCCESSFUL",
        k);
      return 0;
    }
    if (step_res != smt_resultt::P_SATISFIABLE)
    {
      log_error("Solver error in the inductive step at step {}", k);
      return 6;
    }
    close_step(k);
  }
  log_fail("\nVERIFICATION UNKNOWN");
  return 0;
}
