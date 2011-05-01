/*******************************************************************\

Module: Slicer for symex traces

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <hash_cont.h>

#include "slice.h"

/*******************************************************************\

   Class: symex_slicet

 Purpose:

\*******************************************************************/

class symex_slicet
{
public:
  void slice(symex_target_equationt &equation);

protected:
  typedef hash_set_cont<irep_idt, irep_id_hash> symbol_sett;
  
  symbol_sett depends;
  
  void get_symbols(const exprt &expr);
  void get_symbols(const typet &type);

  void slice(symex_target_equationt::SSA_stept &SSA_step);
  void slice_assignment(symex_target_equationt::SSA_stept &SSA_step);
};

/*******************************************************************\

Function: symex_slicet::get_symbols

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void symex_slicet::get_symbols(const exprt &expr)
{
  get_symbols(expr.type());

  forall_operands(it, expr)
    get_symbols(*it);

  if(expr.id()==exprt::symbol)
    depends.insert(expr.get("identifier"));
}

/*******************************************************************\

Function: symex_slicet::get_symbols

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void symex_slicet::get_symbols(const typet &type)
{
}

/*******************************************************************\

Function: symex_slicet::slice

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void symex_slicet::slice(symex_target_equationt &equation)
{
  depends.clear();

  for(symex_target_equationt::SSA_stepst::reverse_iterator
      it=equation.SSA_steps.rbegin();
      it!=equation.SSA_steps.rend();
      it++)
    slice(*it);
}

/*******************************************************************\

Function: symex_slicet::slice

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void symex_slicet::slice(symex_target_equationt::SSA_stept &SSA_step)
{
  get_symbols(SSA_step.guard);

  switch(SSA_step.type)
  {
  case goto_trace_stept::ASSERT:
    get_symbols(SSA_step.cond);
    break;

  case goto_trace_stept::ASSUME:
    get_symbols(SSA_step.cond);
    break;

  case goto_trace_stept::LOCATION:
    // ignore
    break;

  case goto_trace_stept::ASSIGNMENT:
    slice_assignment(SSA_step);
    break;

  case goto_trace_stept::OUTPUT:
    break;

  default:
    assert(false);  
  }
}

/*******************************************************************\

Function: symex_slicet::slice_assignment

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void symex_slicet::slice_assignment(
  symex_target_equationt::SSA_stept &SSA_step)
{
  assert(SSA_step.lhs.id()==exprt::symbol);

  if(depends.find(SSA_step.lhs.get("identifier"))==
     depends.end())
  {
    // we don't really need it
    SSA_step.ignore=true;
  }
  else
    get_symbols(SSA_step.rhs);
}

/*******************************************************************\

Function: slice

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void slice(symex_target_equationt &equation)
{
  symex_slicet symex_slice;
  symex_slice.slice(equation);
}

/*******************************************************************\

Function: simple_slice

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void simple_slice(symex_target_equationt &equation)
{
  // just find the last assertion
  symex_target_equationt::SSA_stepst::iterator
    last_assertion=equation.SSA_steps.end();
  
  for(symex_target_equationt::SSA_stepst::iterator
      it=equation.SSA_steps.begin();
      it!=equation.SSA_steps.end();
      it++)
    if(it->is_assert())
      last_assertion=it;

  // slice away anything after it

  symex_target_equationt::SSA_stepst::iterator s_it=
    last_assertion;

  if(s_it!=equation.SSA_steps.end())
    for(s_it++;
        s_it!=equation.SSA_steps.end();
        s_it++)
      s_it->ignore=true;
}

