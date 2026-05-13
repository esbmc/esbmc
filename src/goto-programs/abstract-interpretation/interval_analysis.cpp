/// \file
/// Interval Analysis

#include "interval_analysis.h"
#include <goto-programs/abstract-interpretation/interval_analysis.h>
#include <goto-programs/abstract-interpretation/interval_domain.h>
#include <unordered_set>
#include <util/prefix.h>
#include <goto-programs/goto_loops.h>
#include <util/time_stopping.h>

template <class Interval>
inline void optimize_expr_interval(expr2tc &expr, const interval_domaint &state)
{
  // Only integers for now (more implementation is needed for floats)
  if (!(is_signedbv_type(expr->type) || is_unsignedbv_type(expr->type) ||
        is_bool_type(expr->type)))
    return;

  // Forward Analysis
  auto interval = state.get_interval<Interval>(expr);

  // Singleton Propagation
  if (interval.singleton() && is_bv_type(expr))
  {
    // Right now we can only do that for bitvectors
    expr = state.make_expression_value<Interval>(interval, expr->type, true);
    return;
  }

  // Boolean intervals
  if (is_bool_type(expr))
  {
    // Expression is always true
    if (!interval.contains(0))
    {
      expr = gen_true_expr();
      return;
    }

    // interval is [0,0] which is always false
    if (interval.singleton())
    {
      expr = gen_false_expr();
      return;
    }
  }
}

static void optimize_expression(expr2tc &expr, const interval_domaint &state)
{
  // Preconditions
  if (is_nil_expr(expr))
    return;

  // We can't simplify addr-of sub-expr.
  // int x = 3; int *ptr = &x; would become int x = 3; int *ptr = &3;
  if (is_address_of2t(expr))
    return;

  // We can't replace the target of an assignment.
  // int x = 3; x = 4; would become int x = 3; 3 = 4;
  if (is_code_assign2t(expr))
  {
    optimize_expression(to_code_assign2t(expr).source, state);
    return;
  }

  // Function calls might have an implicit assignment
  if (is_code_function_call2t(expr))
  {
    optimize_expression(to_code_function_call2t(expr).function, state);
    for (auto &x : to_code_function_call2t(expr).operands)
      optimize_expression(x, state);
    return;
  }

  if (interval_domaint::enable_wrapped_intervals)
    optimize_expr_interval<wrapped_interval>(expr, state);
  else
    optimize_expr_interval<interval_domaint::integer_intervalt>(expr, state);

  // Try sub-expressions
  expr->Foreach_operand(
    [&state](expr2tc &e) -> void { optimize_expression(e, state); });
  simplify(expr);
}

void optimize_function(
  const ait<interval_domaint> &interval_analysis,
  goto_functiont &goto_function)
{
  // Inline optimizations
  Forall_goto_program_instructions (i_it, goto_function.body)
  {
    const interval_domaint &d = interval_analysis[i_it];

    // Singleton Propagation
    optimize_expression(i_it->code, d);
    optimize_expression(i_it->guard, d);
  }
}

/*
 * Instrument an assume containing all the restriction for the set of symbols
 */
inline void instrument_symbol_constraints(
  const ait<interval_domaint> &interval_analysis,
  std::unordered_set<expr2tc, irep2_hash> symbols,
  goto_programt::instructionst::iterator &it,
  goto_functiont &goto_function)
{
  std::vector<expr2tc> symbol_constraints;
  auto state_iterator = interval_analysis.state_map.find(it);
  // We may be trying to instrument an unreachable state
  if (state_iterator == interval_analysis.state_map.end())
    return;
  const interval_domaint &d = state_iterator->second;
  for (const auto &symbol_expr : symbols)
  {
    expr2tc tmp = d.make_expression(symbol_expr);
    if (!is_true(tmp))
      symbol_constraints.push_back(tmp);
  }

  if (!symbol_constraints.empty())
  {
    goto_programt::instructiont instruction;
    instruction.make_assumption(conjunction(symbol_constraints));
    instruction.inductive_step_instruction = config.options.is_kind();
    instruction.location = it->location;
    instruction.function = it->function;
    goto_function.body.insert_swap(it++, instruction);
  }
}

/**
 * Instrument loops with all variables that are affected by it (not only the guards)
 *
 * Before:
 * before-loop
 * 1 : IF !(COND == 0) GOTO 2
 * X = Y // X changes and Y is read-only
 * ...
 * GOTO 1
 * 2: after-loop
 *
 * After:
 * before-loop
 * ASSUME (COND >= some_value && ... && Y <= some_value2)
 * 1: IF !(COND == 0) GOTO 2
 * X = Y // X changes and Y is read-only
 * ...
 * ASSUME (COND >= some_value && ... && Y <= some_value2)
 * GOTO 1
 * 2: ASSUME (COND >= some_value && ... && Y <= some_value2)
 * after-loop
 */
void instrument_loops(
  const ait<interval_domaint> &interval_analysis,
  goto_functionst &program)
{
  Forall_goto_functions (f_it, program)
  {
    if (!f_it->second.body_available)
      continue;

    auto loop = goto_loopst(f_it->first, program, f_it->second);
    for (auto l : loop.get_loops())
    {
      std::unordered_set<expr2tc, irep2_hash> symbols;
      for (auto v : l.get_modified_loop_vars())
        symbols.insert(v);

      for (auto v : l.get_unmodified_loop_vars())
        symbols.insert(v);

      // The before-loop ASSUME is handled by instrument_loop_bounds_after_kind
      // when k-induction is active (it needs to fire AFTER k-induction's
      // havoc, not before). For non-k-induction modes, the loop_head's
      // forward dataflow already constrains the program; emitting a
      // pre-loop assume there would be redundant.

      // Assume #1: just before the back-edge (last body instruction).
      // Tightens the values flowing back into the next iteration's
      // loop_head. instrument_symbol_constraints advances `it` past the
      // (now shifted) back-edge to the first after-loop instruction.
      auto it = l.get_original_loop_exit();
      instrument_symbol_constraints(
        interval_analysis, symbols, it, f_it->second);
      // Assume #2: at the first after-loop instruction. Tightens
      // downstream code that uses the loop's modified vars.
      instrument_symbol_constraints(
        interval_analysis, symbols, it, f_it->second);
    }
  }
}

/// Find the first instruction at or after \p loop_head that was NOT inserted
/// by k-induction's preamble (havoc + entry-condition assume). For loops
/// whose loop_head pointed at an IF before k-induction ran, this returns
/// the (now shifted) IF; for do-while loops it returns the first real body
/// instruction. Stops at \p loop_exit defensively — every preamble
/// instruction is inside the discovered loop, so we should never reach
/// the back-edge while skipping.
static goto_programt::targett skip_inductive_preamble(
  goto_programt::targett loop_head,
  goto_programt::targett loop_exit)
{
  goto_programt::targett it = loop_head;
  while (it != loop_exit && it->inductive_step_instruction)
    ++it;
  return it;
}

/// RAII guard for the process-wide skip_inductive_step_instructions flag,
/// so an exception during the recomputed fixpoint or instrumentation can
/// never leave the flag stuck and silently poison downstream consumers
/// (goto_contractor, any later interval_analysis call).
namespace
{
struct scoped_skip_inductive
{
  scoped_skip_inductive()
  {
    interval_domaint::skip_inductive_step_instructions = true;
  }
  ~scoped_skip_inductive()
  {
    interval_domaint::skip_inductive_step_instructions = false;
  }
};
} // namespace

void instrument_loop_bounds_after_kind(
  goto_functionst &goto_functions,
  const namespacet &ns,
  const optionst &options)
{
  // Recompute the interval fixpoint on the post-k-induction goto-graph.
  // The transparency flag makes the havoc'd assignments and entry-condition
  // assumes no-ops in the transformer, so the state at each loop head
  // matches what it was in the *original* program (before k-induction).
  interval_domaint::set_options(options);
  scoped_skip_inductive _skip_guard;

  ait<interval_domaint> interval_analysis;
  interval_analysis(goto_functions, ns);

  Forall_goto_functions (f_it, goto_functions)
  {
    if (!f_it->second.body_available)
      continue;

    // Walk instructions and find each backwards-GOTO directly, rather than
    // using goto_loopst — the modified-vars analysis it performs is not
    // needed here, and rebuilding it on the post-k-induction graph would
    // misclassify k-induction's inserted assigns.
    for (goto_programt::instructionst::iterator b_it =
           f_it->second.body.instructions.begin();
         b_it != f_it->second.body.instructions.end();
         ++b_it)
    {
      if (!b_it->is_backwards_goto())
        continue;
      if (b_it->targets.size() != 1)
        continue;

      goto_programt::targett loop_head = *b_it->targets.begin();
      if (loop_head == b_it)
        continue; // self-loop; already collapsed by goto_loops

      goto_programt::targett insert_pos =
        skip_inductive_preamble(loop_head, b_it);
      if (insert_pos == b_it)
        continue; // every body instruction is inductive-step — degenerate

      // Collect the variables we want to bound: union of modified and
      // referenced symbols in the loop body. Reuse goto_loopst's logic
      // by rebuilding the loop snapshot — but skip its modified-vars
      // walk and instead gather symbols directly from the body
      // instructions we'll bound.
      std::unordered_set<expr2tc, irep2_hash> symbols;
      for (goto_programt::instructionst::iterator body_it = insert_pos;
           body_it != b_it;
           ++body_it)
      {
        if (body_it->inductive_step_instruction)
          continue;
        get_symbols(body_it->code, symbols);
        get_symbols(body_it->guard, symbols);
      }
      if (symbols.empty())
        continue;

      // Build the bounds expression at the insert position from the
      // (transparency-aware) interval domain state.
      std::vector<expr2tc> symbol_constraints;
      auto state_iterator = interval_analysis.state_map.find(insert_pos);
      if (state_iterator == interval_analysis.state_map.end())
        continue;
      const interval_domaint &d = state_iterator->second;
      for (const auto &symbol_expr : symbols)
      {
        expr2tc tmp = d.make_expression(symbol_expr);
        if (!is_true(tmp))
          symbol_constraints.push_back(tmp);
      }
      if (symbol_constraints.empty())
        continue;

      // Insert via insert_swap so any incoming jumps to insert_pos (e.g.
      // the back-edge after k-induction's chain of insert_swaps) are
      // preserved — they will now land on the ASSUME and fall through to
      // the original instruction.
      goto_programt::instructiont instruction;
      instruction.make_assumption(conjunction(symbol_constraints));
      instruction.inductive_step_instruction = true;
      instruction.location = insert_pos->location;
      instruction.function = insert_pos->function;
      f_it->second.body.insert_swap(insert_pos, instruction);
    }
  }

  goto_functions.update();
}

void instrument_intervals(
  const ait<interval_domaint> &interval_analysis,
  goto_functiont &goto_function,
  const INTERVAL_INSTRUMENTATION_MODE instrument_mode)
{
  assert(instrument_mode != INTERVAL_INSTRUMENTATION_MODE::LOOP_MODE);
  if (!goto_function.body_available)
    return;
  std::unordered_set<expr2tc, irep2_hash> function_symbols;
  Forall_goto_program_instructions (i_it, goto_function.body)
  {
    get_symbols(i_it->code, function_symbols);
    get_symbols(i_it->guard, function_symbols);
  }
  Forall_goto_program_instructions (i_it, goto_function.body)
  {
    std::unordered_set<expr2tc, irep2_hash> local_symbols;
    get_symbols(i_it->code, local_symbols);
    get_symbols(i_it->guard, local_symbols);
    switch (instrument_mode)
    {
    case INTERVAL_INSTRUMENTATION_MODE::NO_INSTRUMENTATION:
    case INTERVAL_INSTRUMENTATION_MODE::LOOP_MODE:
      return;
    case INTERVAL_INSTRUMENTATION_MODE::ALL_INSTRUCTIONS_FULL:
      instrument_symbol_constraints(
        interval_analysis, function_symbols, i_it, goto_function);
      break;
    case INTERVAL_INSTRUMENTATION_MODE::ALL_INSTRUCTIONS_LOCAL:
      instrument_symbol_constraints(
        interval_analysis, local_symbols, i_it, goto_function);
      break;
    case INTERVAL_INSTRUMENTATION_MODE::GUARD_INSTRUCTIONS_FULL:
      if (!(i_it->is_goto() || i_it->is_assume() || i_it->is_assert()))
        continue;
      instrument_symbol_constraints(
        interval_analysis, function_symbols, i_it, goto_function);
      break;
    case INTERVAL_INSTRUMENTATION_MODE::GUARD_INSTRUCTIONS_LOCAL:
      if (!(i_it->is_goto() || i_it->is_assume() || i_it->is_assert()))
        continue;
      instrument_symbol_constraints(
        interval_analysis, local_symbols, i_it, goto_function);
      break;
    }
  }
}

void dump_intervals(
  std::ostringstream &out,
  const goto_functiont &goto_function,
  const ait<interval_domaint> &interval_analysis)
{
  forall_goto_program_instructions (i_it, goto_function.body)
  {
    auto print_vars = [&out, &i_it](const auto &map) {
      for (const auto &interval : map)
      {
        // "state,var,min,max,bot,top";
        out << fmt::format(
          "{},{},{},{},{},{},{},{},{}\n",
          i_it->location_number,
          i_it->location.line().as_string(),
          i_it->location.column().as_string(),
          i_it->location.function().as_string(),
          interval.first,
          (interval.second.lower ? *interval.second.lower : "-inf"),
          (interval.second.upper ? *interval.second.upper : "inf"),
          interval.second.is_bottom(),
          interval.second.is_top());
      }
    };
  }
}

#include <fstream>

void interval_analysis(
  goto_functionst &goto_functions,
  const namespacet &ns,
  const optionst &options,
  const INTERVAL_INSTRUMENTATION_MODE instrument_mode)
{
  fine_timet algorithm_start = current_time();
  // TODO: add options for instrumentation mode
  ait<interval_domaint> interval_analysis;
  interval_domaint::set_options(options);
  interval_analysis(goto_functions, ns);

  if (options.get_bool_option("interval-analysis-dump"))
  {
    std::ostringstream oss;
    interval_analysis.output(goto_functions, oss);
    log_status("{}", oss.str());
  }

  std::string csv_file = options.get_option("interval-analysis-csv-dump");
  if (!csv_file.empty())
  {
    std::ostringstream oss;
    oss << "state,line,column,function,var,min,max,bot,top\n";
    Forall_goto_functions (f_it, goto_functions)
      dump_intervals(oss, f_it->second, interval_analysis);

    std::ofstream csv(csv_file);
    csv << oss.str();
  }

  Forall_goto_functions (f_it, goto_functions)
  {
    optimize_function(interval_analysis, f_it->second);
  }

  if (instrument_mode == INTERVAL_INSTRUMENTATION_MODE::LOOP_MODE)
    instrument_loops(interval_analysis, goto_functions);
  else
  {
    Forall_goto_functions (f_it, goto_functions)
    {
      instrument_intervals(interval_analysis, f_it->second, instrument_mode);
    }
  }
  goto_functions.update();

  fine_timet algorithm_stop = current_time();
  log_status(
    "Interval Analysis time: {}s",
    time2string(algorithm_stop - algorithm_start));
}
