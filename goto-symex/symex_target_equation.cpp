/*******************************************************************\

Module: Symbolic Execution

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <assert.h>

#include <base_type.h>
#include <i2string.h>
#include <std_expr.h>
#include <expr_util.h>

#include <langapi/language_util.h>

#include "goto_symex_state.h"
#include "symex_target_equation.h"

/*******************************************************************\

Function: symex_target_equationt::assignment

  Inputs:

 Outputs:

 Purpose: write to a variable

\*******************************************************************/

void symex_target_equationt::assignment(
  const guardt &guard,
  const exprt &lhs,
  const exprt &original_lhs,
  exprt &rhs,
  const sourcet &source,
  assignment_typet assignment_type)
{
  assert(lhs.is_not_nil());

  SSA_steps.push_back(SSA_stept());
  SSA_stept &SSA_step=SSA_steps.back();

  SSA_step.guard=guard.as_expr();
  SSA_step.lhs=lhs;
  SSA_step.original_lhs=original_lhs;
  SSA_step.rhs.swap(rhs);
  SSA_step.assignment_type=assignment_type;
#if 0
  exprt tmp(guard.as_expr());

  if (guard.is_true())
    SSA_step.cond=equality_exprt(SSA_step.lhs, SSA_step.rhs);
  else
    SSA_step.cond=gen_implies(tmp, equality_exprt(SSA_step.lhs, SSA_step.rhs));
#endif
  SSA_step.cond=equality_exprt(SSA_step.lhs, SSA_step.rhs);
  SSA_step.type=goto_trace_stept::ASSIGNMENT;
  SSA_step.source=source;
}

/*******************************************************************\

Function: symex_target_equationt::location

  Inputs:

 Outputs:

 Purpose: just record a location

\*******************************************************************/

void symex_target_equationt::location(
  const guardt &guard,
  const sourcet &source)
{
  SSA_steps.push_back(SSA_stept());
  SSA_stept &SSA_step=SSA_steps.back();

  SSA_step.guard=guard.as_expr();
  SSA_step.lhs.make_nil();
  SSA_step.type=goto_trace_stept::LOCATION;
  SSA_step.source=source;
}

/*******************************************************************\

Function: symex_target_equationt::trace_event

  Inputs:

 Outputs:

 Purpose: just record output

\*******************************************************************/

void symex_target_equationt::output(
  const guardt &guard,
  const sourcet &source,
  const std::string &fmt,
  const std::list<exprt> &args)
{
  SSA_steps.push_back(SSA_stept());
  SSA_stept &SSA_step=SSA_steps.back();

  SSA_step.guard=guard.as_expr();
  SSA_step.lhs.make_nil();
  SSA_step.type=goto_trace_stept::OUTPUT;
  SSA_step.source=source;
  SSA_step.output_args=args;
  SSA_step.format_string=fmt;
}

/*******************************************************************\

Function: symex_target_equationt::assumption

  Inputs: cond is destroyed

 Outputs:

 Purpose: record an assumption

\*******************************************************************/

void symex_target_equationt::assumption(
  const guardt &guard,
  exprt &cond,
  const sourcet &source)
{
  SSA_steps.push_back(SSA_stept());
  SSA_stept &SSA_step=SSA_steps.back();

  SSA_step.guard=guard.as_expr();
  SSA_step.lhs.make_nil();
  SSA_step.cond.swap(cond);
  SSA_step.type=goto_trace_stept::ASSUME;
  SSA_step.source=source;
}

/*******************************************************************\

Function: symex_target_equationt::assertion

  Inputs: cond is destroyed

 Outputs:

 Purpose: record an assertion

\*******************************************************************/

void symex_target_equationt::assertion(
  const guardt &guard,
  exprt &cond,
  const std::string &msg,
  const sourcet &source)
{
  SSA_steps.push_back(SSA_stept());
  SSA_stept &SSA_step=SSA_steps.back();

  SSA_step.guard=guard.as_expr();
  SSA_step.lhs.make_nil();
  SSA_step.cond.swap(cond);
  SSA_step.type=goto_trace_stept::ASSERT;
  SSA_step.source=source;
  SSA_step.comment=msg;
}

/*******************************************************************\

Function: symex_target_equationt::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void symex_target_equationt::convert(
  prop_convt &prop_conv)
{
  convert_guards(prop_conv);
  convert_assignments(prop_conv);
  convert_assumptions(prop_conv);
  convert_assertions(prop_conv);
  convert_output(prop_conv);
}

/*******************************************************************\

Function: symex_target_equationt::convert_assignments

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void symex_target_equationt::convert_assignments(
  decision_proceduret &decision_procedure) const
{
  for(SSA_stepst::const_iterator it=SSA_steps.begin();
      it!=SSA_steps.end(); it++)
  {
    if(it->is_assignment() && !it->ignore)
    {
      exprt tmp(it->cond);
      ::base_type(tmp, ns);
      decision_procedure.set_to_true(tmp);
    }
  }
}

/*******************************************************************\

Function: symex_target_equationt::convert_guards

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

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
      exprt tmp(it->guard);
      ::base_type(tmp, ns);
      it->guard_literal=prop_conv.convert(tmp);
    }
  }
}

/*******************************************************************\

Function: symex_target_equationt::convert_assumptions

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

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
        exprt tmp(it->cond);
        ::base_type(tmp, ns);
        it->cond_literal=prop_conv.convert(tmp);
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
      exprt tmp(it->cond);
      ::base_type(tmp, ns);

      // do the expression
      literalt tmp_literal=prop_conv.convert(tmp);

      it->cond_literal=prop_conv.prop.limplies(assumption_literal, tmp_literal);

      bv.push_back(prop_conv.prop.lnot(it->cond_literal));
    }
    else if(it->is_assume())
      assumption_literal=
        prop_conv.prop.land(assumption_literal, it->cond_literal);

  if(!bv.empty())
    prop_conv.prop.lcnf(bv);
}

/*******************************************************************\

Function: symex_target_equationt::convert_output

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void symex_target_equationt::convert_output(
  decision_proceduret &dec_proc)
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
        ::base_type(tmp, ns);
        if(tmp.is_constant() ||
           tmp.id()=="string-constant")
          it->converted_output_args.push_back(tmp);
        else
        {
          symbol_exprt symbol;
          symbol.type()=tmp.type();
          symbol.set_identifier("symex::output::"+i2string(output_count++));
          dec_proc.set_to(equality_exprt(tmp, symbol), true);
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
  case goto_trace_stept::LOCATION: out << "LOCATION" << std::endl; break;
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
    out << from_expr(ns, "", cond) << std::endl;

  if(is_assert())
    out << comment << std::endl;

  out << "Guard: " << from_expr(ns, "", guard) << std::endl;
}

bool symex_target_equationt::SSA_stept::operator<(const SSA_stept p2) const
{

  if (type < p2.type)
    return true;
  else if (type != p2.type)
    return false;

  if (guard < p2.guard)
    return true;
  else if (p2.guard < guard)
    return false;

  switch (type) {
  case goto_trace_stept::ASSERT:
  case goto_trace_stept::ASSUME:

    if (cond < p2.cond)
      return true;
    else if (p2.cond < cond)
      return false;

    if (comment < p2.comment)
      return true;
    return false;

  case goto_trace_stept::OUTPUT:

    if (format_string < p2.format_string)
      return true;
    else if (p2.format_string < format_string)
      return false;

    /* So the format string is the same, how on earth does one compare two
     * lists? XXX - do this in the future. Shouldn't affect any operation for
     * now */
    return false;

  case goto_trace_stept::LOCATION:

    if (source < p2.source)
      return true;
    else if (p2.source < source)
      return false;

    return false; /* XXX - what else? */

  case goto_trace_stept::ASSIGNMENT:

    if (lhs < p2.lhs)
      return true;
    else if (p2.lhs < lhs)
      return false;

    if (rhs < p2.rhs)
      return true;
    else if (p2.rhs < rhs)
      return false;

    if (original_lhs < p2.original_lhs)
      return true;
    else if (p2.original_lhs < original_lhs)
      return false;

    if (assignment_type < p2.assignment_type)
      return true;
    else if (p2.assignment_type < assignment_type)
      return false;

    return false;
  }

  return false;
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

#include <langapi/mode.h>
#include <langapi/languages.h>
extern "C" {
#include <openssl/sha.h>
#include <string.h>
#include <stdint.h>
};

static std::string state_to_ignore[8] = 
{"\\guard_exec", "trds_count", "trds_in_run", "deadlock_wait", "deadlock_mutex",
"count_lock", "count_wait", "unlocked"};

symex_target_equationt::equation_hash symex_target_equationt::generate_hash(namespacet ns) const
{
  std::set<SSA_stept> steps;
  std::string serialised;

  languagest languages(ns, MODE_C);

  /* Put all steps in a set: IE, order them */
  for(symex_target_equationt::SSA_stepst::const_iterator
      it=SSA_steps.begin();
      it!=SSA_steps.end(); it++) {
    if (it->is_assignment() && it->assignment_type == HIDDEN)
      continue;

    steps.insert(*it);
  }

  for(std::set<SSA_stept>::const_iterator
      it=steps.begin();
      it!=steps.end(); it++)
  {
    if(it->is_assignment())
    {
      int i;
      std::string string_value;

      languages.from_expr(it->cond, string_value);

      for (i = 0; i < 8; i++) {
        if (string_value.find(state_to_ignore[i]) != std::string::npos)
          break;
      }

      if (i != 8)
        continue;

      serialised += string_value;
      serialised += "\n";
    }
  }

  std::cout << "ohai:\n" << serialised << std::endl;

  uint8_t out[32];
  const char *str = serialised.c_str();
  SHA256_CTX c;
  SHA256_Init(&c);
  SHA256_Update(&c, str, strlen(str));
  SHA256_Final(out, &c);
  equation_hash e(out);
  std::cout << "with hash:\n" << e.to_string();
  return e;
}
