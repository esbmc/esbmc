/// \file
/// Interval Analysis

#include <goto-programs/abstract-interpretation/interval_analysis.h>
#include <goto-programs/abstract-interpretation/interval_domain.h>
#include <unordered_set>
#include <util/prefix.h>

static inline void get_symbols(
  const expr2tc &expr,
  std::unordered_set<expr2tc, irep2_hash> &symbols)
{
  if(is_nil_expr(expr))
    return;

  if(is_symbol2t(expr))
  {
    symbol2t s = to_symbol2t(expr);
    if(s.thename.as_string().find("__ESBMC_") != std::string::npos)
      return;
    symbols.insert(expr);
  }

  expr->foreach_operand(
    [&symbols](const expr2tc &e) -> void { get_symbols(e, symbols); });
}

static void optimize_expression(expr2tc &expr, const interval_domaint &state)
{
  // Preconditions
  if(is_nil_expr(expr))
    return;

  // We can't simplify addr-of sub-expr.
  // int x = 3; int *ptr = &x; would become int x = 3; int *ptr = &3;
  if(is_address_of2t(expr))
    return;

  // We can't replace the target of an assignment.
  // int x = 3; x = 4; would become int x = 3; 3 = 4;
  if(is_code_assign2t(expr))
  {
    optimize_expression(to_code_assign2t(expr).source, state);
    return;
  }

  // Function calls might have an implicit assignment
  if(is_code_function_call2t(expr))
  {
    for(auto &x : to_code_function_call2t(expr).operands)
      optimize_expression(x, state);
    return;
  }

  // Forward Analysis
  auto interval = state.get_interval<integer_intervalt>(expr);

  // Singleton Propagation
  if(interval.singleton() && is_bv_type(expr))
  {
    // Right now we can only do that for bitvectors (more implementation is needed for floats)
    expr = state.make_expression_value<integer_intervalt>(
      interval, expr->type, true);
    return;
  }

  // Boolean intervals
  if(is_bool_type(expr))
  {
    // Expression is always true
    if(!interval.contains(0))
    {
      expr = gen_true_expr();
      return;
    }

    // interval is [0,0] which is always false
    if(interval.singleton())
    {
      expr = gen_false_expr();
      return;
    }
  }

  // Try sub-expressions
  expr->Foreach_operand(
    [&state](expr2tc &e) -> void { optimize_expression(e, state); });
  simplify(expr);
}

void instrument_intervals(
  const ait<interval_domaint> &interval_analysis,
  goto_functiont &goto_function)
{
  std::unordered_set<expr2tc, irep2_hash> symbols;

  // Inline optimizations
  Forall_goto_program_instructions(i_it, goto_function.body)
  {
    const interval_domaint &d = interval_analysis[i_it];

    // Singleton Propagation
    optimize_expression(i_it->code, d);
    optimize_expression(i_it->guard, d);

    get_symbols(i_it->code, symbols);
    get_symbols(i_it->guard, symbols);
  }

  // Instrumentation of assumptions
  Forall_goto_program_instructions(i_it, goto_function.body)
  {
    if(i_it == goto_function.body.instructions.begin())
    {
      // first instruction, we instrument
    }
    else
    {
      if(i_it->is_assume() || i_it->is_assert() || i_it->is_goto())
      {
        // We may be able to simplify here
        const interval_domaint &d = interval_analysis[i_it];

        // Evaluate the simplified expression
        tvt guard = interval_domaint::eval_boolean_expression(i_it->guard, d);
        if(i_it->is_goto())
          guard = !guard;
        // If guard is always true... convert it into a skip!
        if(guard.is_true() && interval_domaint::enable_assertion_simplification)
          i_it->make_skip();
        // If guard is always false... convert it to trivial!
        if(
          guard.is_false() && interval_domaint::enable_assertion_simplification)
          i_it->guard = i_it->is_goto() ? gen_true_expr() : gen_false_expr();

        // Let's instrument an assumption with symbols that affect the guard
        std::vector<expr2tc> assumption;
        std::unordered_set<expr2tc, irep2_hash> guard_symbols;
        get_symbols(i_it->guard, guard_symbols);
        for(const auto &symbol_expr : guard_symbols)
        {
          expr2tc tmp = d.make_expression(symbol_expr);
          if(!is_true(tmp))
            assumption.push_back(tmp);
        }

        if(!assumption.empty())
        {
          goto_programt::targett t = goto_function.body.insert(i_it);
          t->make_assumption(conjunction(assumption));
          t->inductive_step_instruction = config.options.is_kind();
        }

        continue;
      }

      /**
       * The instrumentation of the assume will happen in:
       *
       * 1. After IF (and)
       *  IF !(a > 42) GOTO 5
       *    +++ ASSUME (a > 42)
       *
       * 2. After a function call
       *  FUCTION_CALL(FOO)
       *    +++ ASSUME(state-after-foo)
       *
       * 3. Before a function call
       *  +++ ASSUME(state-before-foo)
       *  FUCNTION_CALL(FOO)
       *
       * 4. Before a target
       *  +++ ASSUME(current-state)
       *  1: ....
      */
      goto_programt::const_targett previous = i_it;
      previous--;
      if(previous->is_goto() && !is_true(previous->guard))
      {
        // we follow a branch, instrument
      }
      else if(previous->is_function_call() && !is_true(previous->guard))
      {
        // we follow a function call, instrument
      }
      else if(i_it->is_target() || i_it->is_function_call())
      {
        // we are a target or a function call, instrument
      }
      else
        continue; // don't instrument
    }

    const interval_domaint &d = interval_analysis[i_it];
    std::vector<expr2tc> assertion;
    for(const auto &symbol_expr : symbols)
    {
      expr2tc tmp = d.make_expression(symbol_expr);
      if(!is_true(tmp))
        assertion.push_back(tmp);
    }

    if(!assertion.empty())
    {
      goto_programt::targett t = goto_function.body.insert(i_it);
      t->make_assumption(conjunction(assertion));
      t->inductive_step_instruction = config.options.is_kind();
#if 0
      // TODO: This is crashing cases like
      // email_spec11_productSimulator_false-unreach-call_true-termination.cil.c
      i_it++; // goes to original instruction
      t->location = i_it->location;
      t->function = i_it->function;
#endif
    }
  }
}

void dump_intervals(
  std::ostringstream &out,
  const goto_functiont &goto_function,
  const ait<interval_domaint> &interval_analysis)
{
  forall_goto_program_instructions(i_it, goto_function.body)
  {
    const interval_domaint &d = interval_analysis[i_it];
    auto print_vars = [&out, &i_it](const auto &map) {
      for(const auto &interval : map)
      {
        // "state,var,min,max,bot,top";
        out << fmt::format(
          "{},{},{},{},{},{},{},{},{}\n",
          i_it->location_number,
          i_it->location.line().as_string(),
          i_it->location.column().as_string(),
          i_it->location.function().as_string(),
          interval.first,
          (interval.second.lower_set ? interval.second.lower : "-inf"),
          (interval.second.upper_set ? interval.second.upper : "inf"),
          interval.second.is_bottom(),
          interval.second.is_top());
      }
    };
    d.enable_wrapped_intervals ? print_vars(d.get_wrap_map())
                               : print_vars(d.get_int_map());
  }
}

#include <fstream>

void interval_analysis(
  goto_functionst &goto_functions,
  const namespacet &ns,
  const optionst &options)
{
  ait<interval_domaint> interval_analysis;
  interval_domaint::set_options(options);
  interval_analysis(goto_functions, ns);

  if(options.get_bool_option("interval-analysis-dump"))
  {
    std::ostringstream oss;
    interval_analysis.output(goto_functions, oss);
    log_status("{}", oss.str());
  }

  std::string csv_file = options.get_option("interval-analysis-csv-dump");
  if(!csv_file.empty())
  {
    std::ostringstream oss;
    oss << "state,line,column,function,var,min,max,bot,top\n";
    Forall_goto_functions(f_it, goto_functions)
      dump_intervals(oss, f_it->second, interval_analysis);

    std::ofstream csv(csv_file);
    csv << oss.str();
  }

  Forall_goto_functions(f_it, goto_functions)
    instrument_intervals(interval_analysis, f_it->second);

  goto_functions.update();
}
