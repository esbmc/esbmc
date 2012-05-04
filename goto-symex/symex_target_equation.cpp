/*******************************************************************\

Module: Symbolic Execution

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <irep2.h>
#include <migrate.h>
#include <assert.h>

#include <i2string.h>
#include <std_expr.h>
#include <expr_util.h>
#include <irep2.h>
#include <migrate.h>

#include <langapi/language_util.h>

#include "goto_symex_state.h"
#include "symex_target_equation.h"

void symex_target_equationt::assignment(
  const expr2tc &guard,
  const expr2tc &lhs,
  const expr2tc &original_lhs,
  const expr2tc &rhs,
  const sourcet &source,
  std::vector<dstring> stack_trace,
  assignment_typet assignment_type)
{
  assert(!is_nil_expr(lhs));

  SSA_steps.push_back(SSA_stept());
  SSA_stept &SSA_step=SSA_steps.back();

  SSA_step.guard = guard;
  SSA_step.lhs = lhs;
  SSA_step.original_lhs = original_lhs;
  SSA_step.rhs = rhs;
  SSA_step.guard = guard;
  SSA_step.lhs = lhs;
  SSA_step.original_lhs = original_lhs;
  SSA_step.rhs = rhs;
  SSA_step.assignment_type=assignment_type;
  SSA_step.cond = expr2tc(new equality2t(lhs, rhs));
  SSA_step.type=goto_trace_stept::ASSIGNMENT;
  SSA_step.source=source;
  SSA_step.stack_trace = stack_trace;
}

void symex_target_equationt::output(
  const expr2tc &guard,
  const sourcet &source,
  const std::string &fmt,
  const std::list<exprt> &args)
{
  SSA_steps.push_back(SSA_stept());
  SSA_stept &SSA_step=SSA_steps.back();

  SSA_step.guard = guard;
  SSA_step.type=goto_trace_stept::OUTPUT;
  SSA_step.source=source;
  SSA_step.output_args=args;
  SSA_step.format_string=fmt;
}

void symex_target_equationt::assumption(
  const expr2tc &guard,
  const expr2tc &cond,
  const sourcet &source)
{
  SSA_steps.push_back(SSA_stept());
  SSA_stept &SSA_step=SSA_steps.back();

  SSA_step.guard = guard;
  SSA_step.cond = cond;
  SSA_step.type=goto_trace_stept::ASSUME;
  SSA_step.source=source;
}

void symex_target_equationt::assertion(
  const guardt &guard,
  exprt &cond,
  const std::string &msg,
  std::vector<dstring> stack_trace,
  const sourcet &source)
{
  SSA_steps.push_back(SSA_stept());
  SSA_stept &SSA_step=SSA_steps.back();

  expr2tc new_guard, new_cond;
  migrate_expr(guard.as_expr(), new_guard);
  migrate_expr(cond, new_cond);

  SSA_step.guard = new_guard;
  SSA_step.cond = new_cond;
  SSA_step.type=goto_trace_stept::ASSERT;
  SSA_step.source=source;
  SSA_step.comment=msg;
  SSA_step.stack_trace = stack_trace;
}

void symex_target_equationt::convert(
  prop_convt &prop_conv)
{
  convert_guards(prop_conv);
  convert_assignments(prop_conv);
  convert_assumptions(prop_conv);
  convert_assertions(prop_conv);
  convert_output(prop_conv);
}

void symex_target_equationt::convert_assignments(prop_convt &prop_conv) const
{
  for(SSA_stepst::const_iterator it=SSA_steps.begin();
      it!=SSA_steps.end(); it++)
  {
    if(it->is_assignment() && !it->ignore)
    {
      prop_conv.set_to(it->cond, true);
    }
  }
}

void symex_target_equationt::convert_guards(
  prop_convt &prop_conv)
{
  for(SSA_stepst::iterator it=SSA_steps.begin();
      it!=SSA_steps.end(); it++)
  {
    if(it->ignore)
      it->guard_literal=const_literal(false);
    else
    {
      it->guard_literal=prop_conv.convert(it->guard);
    }
  }
}

void symex_target_equationt::convert_assumptions(
  prop_convt &prop_conv)
{
  for(SSA_stepst::iterator it=SSA_steps.begin();
      it!=SSA_steps.end(); it++)
  {
    if(it->is_assume())
    {
      if(it->ignore)
        it->cond_literal=const_literal(true);
      else
      {
        it->cond_literal=prop_conv.convert(it->cond);
      }
    }
  }
}

/*******************************************************************\

Function: symex_target_equationt::convert_assertions

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void symex_target_equationt::convert_assertions(
  prop_convt &prop_conv)
{
  bvt bv;

  bv.reserve(SSA_steps.size());

  literalt assumption_literal=const_literal(true);

  for(SSA_stepst::iterator it=SSA_steps.begin();
      it!=SSA_steps.end(); it++)
    if(it->is_assert())
    {

      // do the expression
      literalt tmp_literal=prop_conv.convert(it->cond);

      it->cond_literal=prop_conv.limplies(assumption_literal, tmp_literal);

      bv.push_back(prop_conv.lnot(it->cond_literal));
    }
    else if(it->is_assume())
      assumption_literal=
        prop_conv.land(assumption_literal, it->cond_literal);

  if(!bv.empty())
    prop_conv.lcnf(bv);
}

/*******************************************************************\

Function: symex_target_equationt::convert_output

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void symex_target_equationt::convert_output(prop_convt &prop_conv)
{
  unsigned output_count=0;

  for(SSA_stepst::iterator it=SSA_steps.begin();
      it!=SSA_steps.end(); it++)
    if(it->is_output() && !it->ignore)
    {
      for(std::list<exprt>::const_iterator
          o_it=it->output_args.begin();
          o_it!=it->output_args.end();
          o_it++)
      {
        exprt tmp=*o_it;
        if(tmp.is_constant() ||
           tmp.id()=="string-constant")
          it->converted_output_args.push_back(tmp);
        else
        {
          symbol_exprt symbol;
          symbol.type()=tmp.type();
          symbol.set_identifier("symex::output::"+i2string(output_count++));
          expr2tc new_expr;
          migrate_expr(equality_exprt(tmp, symbol), new_expr);
          prop_conv.set_to(new_expr, true);
          it->converted_output_args.push_back(symbol);
        }
      }
    }
}

/*******************************************************************\

Function: symex_target_equationt::output

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void symex_target_equationt::output(std::ostream &out) const
{
  for(SSA_stepst::const_iterator
      it=SSA_steps.begin();
      it!=SSA_steps.end();
      it++)
  {
    it->output(ns, out);
    out << "--------------" << std::endl;
  }
}

/*******************************************************************\

Function: symex_target_equationt::SSA_stept::output

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void symex_target_equationt::SSA_stept::output(
  const namespacet &ns,
  std::ostream &out) const
{
  if(source.is_set)
  {
    out << "Thread " << source.thread_nr;

    if(source.pc->location.is_not_nil())
      out << " " << source.pc->location << std::endl;
    else
      out << std::endl;
  }

  switch(type)
  {
  case goto_trace_stept::ASSERT: out << "ASSERT" << std::endl; break;
  case goto_trace_stept::ASSUME: out << "ASSUME" << std::endl; break;
  case goto_trace_stept::OUTPUT: out << "OUTPUT" << std::endl; break;

  case goto_trace_stept::ASSIGNMENT:
    out << "ASSIGNMENT (";
    switch(assignment_type)
    {
    case HIDDEN: out << "HIDDEN"; break;
    case STATE: out << "STATE"; break;
    default:;
    }

    out << ")" << std::endl;
    break;

  default: assert(false);
  }

  if(is_assert() || is_assume() || is_assignment())
    out << from_expr(ns, "", migrate_expr_back(cond)) << std::endl;

  if(is_assert())
    out << comment << std::endl;

  out << "Guard: " << from_expr(ns, "", migrate_expr_back(guard)) << std::endl;
}

/*******************************************************************\

Function: operator <<

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::ostream &operator<<(
  std::ostream &out,
  const symex_target_equationt &equation)
{
  equation.output(out);
  return out;
}
