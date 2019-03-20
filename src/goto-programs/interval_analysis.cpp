/*******************************************************************\

Module: Interval Analysis

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

/// \file
/// Interval Analysis

#include <goto-programs/interval_analysis.h>
#include <goto-programs/interval_domain.h>

static inline void
get_symbols(const expr2tc &expr, hash_set_cont<expr2tc, irep2_hash> &symbols)
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

void instrument_intervals(
  const ait<interval_domaint> &interval_analysis,
  goto_functiont &goto_function)
{
  hash_set_cont<expr2tc, irep2_hash> symbols;

  forall_goto_program_instructions(i_it, goto_function.body)
  {
    get_symbols(i_it->code, symbols);
    get_symbols(i_it->guard, symbols);
  }

  Forall_goto_program_instructions(i_it, goto_function.body)
  {
    if(i_it == goto_function.body.instructions.begin())
    {
      // first instruction, we instrument
    }
    else
    {
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
      t->inductive_step_instruction = true;
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

void interval_analysis(goto_functionst &goto_functions, const namespacet &ns)
{
  ait<interval_domaint> interval_analysis;

  interval_analysis(goto_functions, ns);

  Forall_goto_functions(f_it, goto_functions)
    instrument_intervals(interval_analysis, f_it->second);

  goto_functions.update();
}
