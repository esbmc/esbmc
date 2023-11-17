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
  if (is_nil_expr(expr))
    return;

  if (is_symbol2t(expr))
  {
    symbol2t s = to_symbol2t(expr);
    if (s.thename.as_string().find("__ESBMC_") != std::string::npos)
      return;
    symbols.insert(expr);
  }

  expr->foreach_operand(
    [&symbols](const expr2tc &e) -> void { get_symbols(e, symbols); });
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
    for (auto &x : to_code_function_call2t(expr).operands)
      optimize_expression(x, state);
    return;
  }

  // Forward Analysis
  auto interval = state.get_interval<integer_intervalt>(expr);

  // Singleton Propagation
  if (interval.singleton() && is_bv_type(expr))
  {
    // Right now we can only do that for bitvectors (more implementation is needed for floats)
    expr = state.make_expression_value<integer_intervalt>(
      interval, expr->type, true);
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

  // Try sub-expressions
  expr->Foreach_operand(
    [&state](expr2tc &e) -> void { optimize_expression(e, state); });
  simplify(expr);
}

void instrument_intervals(
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

    // TODO: Move Guard Simplification to here
  }

  // Instrumentation of assumptions
  Forall_goto_program_instructions (i_it, goto_function.body)
  {
    if (!(i_it->is_goto() || i_it->is_assume() || i_it->is_assert()))
      continue;

    // Let's instrument everything that affect the current instruction
    std::unordered_set<expr2tc, irep2_hash> symbols;
    get_symbols(i_it->code, symbols);
    get_symbols(i_it->guard, symbols);

    if (!symbols.size())
      continue;

    const interval_domaint &d = interval_analysis[i_it];
    std::vector<expr2tc> symbol_constraints;
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
      instruction.location = i_it->location;
      instruction.function = i_it->function;
      goto_function.body.insert_swap(i_it++, instruction);
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
    const interval_domaint &d = interval_analysis[i_it];
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
    instrument_intervals(interval_analysis, f_it->second);

  goto_functions.update();
}
