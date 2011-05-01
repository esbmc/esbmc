/*******************************************************************\

Module: Symbolic Execution of ANSI-C

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <pointer-analysis/dereference.h>
#include <langapi/language_util.h>

#include "goto_symex.h"
#include "renaming_ns.h"

/*******************************************************************\

   Class: symex_dereference_statet

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

class symex_dereference_statet:
  public dereference_callbackt
{
public:
  symex_dereference_statet(
    goto_symext &_goto_symex,
    goto_symext::statet &_state):
    goto_symex(_goto_symex),
    state(_state)
  {
  }

protected:
  goto_symext &goto_symex;
  goto_symext::statet &state;

  // overloads from dereference_callbackt
  virtual bool is_valid_object(const irep_idt &identifier)
  {
    return true;
  }
#if 1
  virtual void dereference_failure(
    const std::string &property,
    const std::string &msg,
    const guardt &guard);
#endif
  virtual void get_value_set(
    const exprt &expr,
    value_setst::valuest &value_set);

  virtual bool has_failed_symbol(
    const exprt &expr,
    const symbolt *&symbol);
};

/*******************************************************************\

Function: symex_dereference_statet::dereference_failure

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void symex_dereference_statet::dereference_failure(
  const std::string &property,
  const std::string &msg,
  const guardt &guard)
{
}

/*******************************************************************\

Function: symex_dereference_statet::has_failed_symbol

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool symex_dereference_statet::has_failed_symbol(
  const exprt &expr,
  const symbolt *&symbol)
{
  renaming_nst renaming_ns(goto_symex.ns, state);

  if(expr.id()==exprt::symbol)
  {
    const symbolt &ptr_symbol=
      renaming_ns.lookup(expr.get("identifier"));

    const irep_idt &failed_symbol=
      ptr_symbol.type.get("#failed_symbol");

    if(failed_symbol=="") return false;

    return !renaming_ns.lookup(failed_symbol, symbol);
  }

  return false;
}

/*******************************************************************\

Function: symex_dereference_statet::get_value_set

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void symex_dereference_statet::get_value_set(
  const exprt &expr,
  value_setst::valuest &value_set)
{
  renaming_nst renaming_ns(goto_symex.ns, state);

  state.value_set.get_value_set(expr, value_set, renaming_ns);

  #if 0
  std::cout << "**************************\n";
  state.value_set.output(std::cout, renaming_ns);
  std::cout << "**************************\n";
  #endif

  #if 0
  std::cout << "E: " << expr.pretty() << std::endl;
  #endif

  #if 0
  std::cout << "**************************\n";
  for(expr_sett::const_iterator it=value_set.begin();
      it!=value_set.end();
      it++)
    std::cout << from_expr(renaming_ns, "", *it) << std::endl;
  std::cout << "**************************\n";
  #endif
}

/*******************************************************************\

Function: goto_symext::dereference_rec

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symext::dereference_rec(
  exprt &expr,
  guardt &guard,
  dereferencet &dereference,
  const bool write)
{
  if(expr.id()==exprt::deref ||
     expr.id()=="implicit_dereference")
  {
    if(expr.operands().size()!=1)
      throw "dereference takes one operand";

    exprt tmp;
    tmp.swap(expr.op0());

    // first make sure there are no dereferences in there
    dereference_rec(tmp, guard, dereference, false);

    dereference.dereference(tmp, guard, write?dereferencet::WRITE:dereferencet::READ);
    expr.swap(tmp);
  }
  else if(expr.id()==exprt::index &&
          expr.operands().size()==2 &&
          expr.op0().type().id()=="pointer")
  {
    exprt tmp(exprt::plus, expr.op0().type());
    tmp.operands().swap(expr.operands());

    // first make sure there are no dereferences in there
    dereference_rec(tmp, guard, dereference, false);

    dereference.dereference(tmp, guard, write?dereferencet::WRITE:dereferencet::READ);
    tmp.swap(expr);
  }
  else
  {
    Forall_operands(it, expr)
      dereference_rec(*it, guard, dereference, write);
  }
}

/*******************************************************************\

Function: goto_symext::dereference

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symext::dereference(
  exprt &expr,
  statet &state,
  const bool write,
  unsigned node_id)
{
  symex_dereference_statet symex_dereference_state(*this, state);
  renaming_nst renaming_ns(ns, state);

  dereferencet dereference(
    renaming_ns,
    new_context,
    options,
    symex_dereference_state);

  // needs to be renamed to level 1
  assert(!state.call_stack.empty());
  state.top().level1.rename(expr,node_id);

  guardt guard;
  dereference_rec(expr, guard, dereference, write);
}
