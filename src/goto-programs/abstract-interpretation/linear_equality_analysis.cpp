/// \file
/// Linear Equality Analysis
///
/// Algorithm:
///   1. Run ait<linear_equality_domaint> to compute a fixedpoint of linear
///      equality states over the full GOTO control-flow graph.
///   2. For each loop, collect the variables modified or read inside it.
///   3. At the loop body entry (the instruction after the loop-condition IF),
///      insert ASSUME(invariant) where invariant is the conjunction of all
///      inferred equalities restricted to loop-relevant variables.
///      The ASSUME is tagged with inductive_step_instruction so that k-induction
///      can exploit it when reasoning about arbitrarily many loop iterations.

#include <goto-programs/abstract-interpretation/linear_equality_analysis.h>

#include <goto-programs/abstract-interpretation/linear_equality_domain.h>
#include <goto-programs/goto_loops.h>
#include <irep2/irep2_utils.h>
#include <util/dstring.h>
#include <util/message.h>
#include <util/time_stopping.h>

#include <sstream>
#include <unordered_set>

using loop_var_set = std::unordered_set<irep_idt, dstring_hash>;

static loop_var_set collect_loop_vars(const loopst &loop)
{
  loop_var_set vars;
  for (const auto &v : loop.get_modified_loop_vars())
    if (is_symbol2t(v))
      vars.insert(to_symbol2t(v).thename);
  for (const auto &v : loop.get_unmodified_loop_vars())
    if (is_symbol2t(v))
      vars.insert(to_symbol2t(v).thename);
  return vars;
}

static void instrument_loop(
  const linear_equality_analysist &analysis,
  const loopst &loop,
  const loop_var_set &loop_vars,
  goto_functiont &goto_function)
{
  auto head_it = loop.get_original_loop_head();
  auto state_it = analysis.state_map.find(head_it);
  if (state_it == analysis.state_map.end())
    return;

  const linear_equality_domaint &state = state_it->second;
  if (state.is_bottom() || state.is_top())
    return;

  expr2tc invariant = state.to_predicate(loop_vars);
  if (is_true(invariant))
    return;

  // head_it points to the loop-condition IF; insert after it so the ASSUME
  // lands at the body entry and is assumed on every iteration.
  auto body_it = std::next(head_it);
  goto_programt::instructiont assume_insn;
  assume_insn.make_assumption(invariant);
  assume_insn.inductive_step_instruction = config.options.is_kind();
  assume_insn.location = body_it->location;
  assume_insn.function = body_it->function;
  goto_function.body.insert_swap(body_it, assume_insn);
}

void linear_equality_analysis(
  goto_functionst &goto_functions,
  const namespacet &ns,
  const optionst &options)
{
  fine_timet t_start = current_time();

  linear_equality_analysist lin_analysis;
  lin_analysis(goto_functions, ns);

  const bool dump = options.get_bool_option("linear-equality-analysis-dump");
  std::ostringstream oss;
  if (dump)
    oss << "=== Linear Equality Analysis: inferred loop invariants ===\n";

  Forall_goto_functions (f_it, goto_functions)
  {
    if (!f_it->second.body_available)
      continue;

    goto_loopst loops(f_it->first, goto_functions, f_it->second);
    unsigned loop_num = 0;

    for (auto &loop : loops.get_loops())
    {
      loop_var_set vars = collect_loop_vars(loop);

      if (dump)
      {
        auto head_it = loop.get_original_loop_head();
        auto state_it = lin_analysis.state_map.find(head_it);
        if (state_it != lin_analysis.state_map.end())
        {
          const linear_equality_domaint &state = state_it->second;
          bool first = true;
          for (const auto &kv : *state.equations)
          {
            if (!vars.count(kv.first))
              continue;
            if (first)
            {
              oss << "Function " << f_it->first << ", loop " << loop_num << " ("
                  << head_it->location.as_string() << "):\n";
              first = false;
            }
            oss << "  " << kv.first << " == " << kv.second.to_string() << "\n";
          }
        }
      }

      instrument_loop(lin_analysis, loop, vars, f_it->second);
      ++loop_num;
    }
  }

  if (dump)
    log_status("{}", oss.str());

  goto_functions.update();
  log_status(
    "Linear Equality Analysis time: {}s",
    time2string(current_time() - t_start));
}
