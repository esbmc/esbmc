/*******************************************************************\

Module: Slicer for symex traces

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <goto-symex/slice.h>

symex_slicet::symex_slicet() : ignored(0), single_slice(false)
{
}

void symex_slicet::get_symbols(const expr2tc &expr)
{

  expr->foreach_operand([this] (const expr2tc &e) {
    if (!is_nil_expr(e))
      get_symbols(e);
    }
  );

  if (is_symbol2t(expr)) {
    const symbol2t &tmp = to_symbol2t(expr);
    depends.insert(tmp.get_symbol_name());
  }
}

void symex_slicet::slice(boost::shared_ptr<symex_target_equationt> &eq)
{
  depends.clear();

  for(symex_target_equationt::SSA_stepst::reverse_iterator
      it = eq->SSA_steps.rbegin();
      it != eq->SSA_steps.rend();
      it++)
    slice(*it);
}

void symex_slicet::slice_for_symbols(
  boost::shared_ptr<symex_target_equationt> &eq,
  const expr2tc &expr)
{
  get_symbols(expr);
  single_slice = true;

  for(symex_target_equationt::SSA_stepst::reverse_iterator
      it = eq->SSA_steps.rbegin();
      it != eq->SSA_steps.rend();
      it++)
    slice(*it);
}

void symex_slicet::slice(symex_target_equationt::SSA_stept &SSA_step)
{
  if (!single_slice)
    get_symbols(SSA_step.guard);

  switch(SSA_step.type)
  {
  case goto_trace_stept::ASSERT:
    if (!single_slice)
      get_symbols(SSA_step.cond);
    break;

  case goto_trace_stept::ASSUME:
    if (!single_slice)
      get_symbols(SSA_step.cond);
    break;

  case goto_trace_stept::ASSIGNMENT:
    slice_assignment(SSA_step);
    break;

  case goto_trace_stept::OUTPUT:
    break;

  case goto_trace_stept::RENUMBER:
    slice_renumber(SSA_step);
    break;

  default:
    assert(false);
  }
}

void symex_slicet::slice_assignment(
  symex_target_equationt::SSA_stept &SSA_step)
{
  assert(is_symbol2t(SSA_step.lhs));

  const symbol2t &tmp = to_symbol2t(SSA_step.lhs);
  if (depends.find(tmp.get_symbol_name()) == depends.end())
  {
    // we don't really need it
    SSA_step.ignore=true;
    ignored++;
  }
  else
  {
    get_symbols(SSA_step.rhs);
    // Remove this symbol as we won't be seeing any references to it further
    // into the history.
    depends.erase(tmp.get_symbol_name());
  }
}

void symex_slicet::slice_renumber(
  symex_target_equationt::SSA_stept &SSA_step)
{
  assert(is_symbol2t(SSA_step.lhs));

  if (depends.find(to_symbol2t(SSA_step.lhs).get_symbol_name())
              == depends.end())
  {
    // we don't really need it
    SSA_step.ignore=true;
    ignored++;
  }

  // Don't collect the symbol; this insn has no effect on dependencies.
}

u_int64_t slice(boost::shared_ptr<symex_target_equationt> &eq)
{
  symex_slicet symex_slice;
  symex_slice.slice(eq);
  return symex_slice.ignored;
}

u_int64_t simple_slice(boost::shared_ptr<symex_target_equationt> &eq)
{
  u_int64_t ignored = 0;

  // just find the last assertion
  symex_target_equationt::SSA_stepst::iterator
    last_assertion = eq->SSA_steps.end();

  for(symex_target_equationt::SSA_stepst::iterator
      it = eq->SSA_steps.begin();
      it != eq->SSA_steps.end();
      it++)
    if(it->is_assert())
      last_assertion=it;

  // slice away anything after it

  symex_target_equationt::SSA_stepst::iterator s_it=
    last_assertion;

  if(s_it != eq->SSA_steps.end())
    for(s_it++;
        s_it!= eq->SSA_steps.end();
        s_it++)
    {
      s_it->ignore=true;
      ignored++;
    }

  return ignored;
}
