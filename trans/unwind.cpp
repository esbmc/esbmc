/*******************************************************************\

Module: Unwinding Transition Systems

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <namespace.h>
#include <i2string.h>
#include <find_symbols.h>
#include <expr_util.h>

#include "instantiate.h"
#include "unwind.h"

/*******************************************************************\

Function: unwind

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void unwind(
  const transt &trans,
  messaget &message,
  propt &solver,
  bmc_mapt &map,
  const namespacet &ns)
{
  const exprt &op_invar=trans.invar();
  const exprt &op_init=trans.init();
  const exprt &op_trans=trans.trans();

  // general constraints

  if(!op_invar.is_true())
  {
    message.status("General constraints");

    for(unsigned c=0; c<map.get_no_timeframes(); c++)
      instantiate_constraint(solver, map, op_invar, c, c+1, ns, message);
  }

  // initial state

  if(!op_init.is_true())
  {
    message.status("Initial state");

    instantiate_constraint(solver, map, op_init, 0, 1, ns, message);
  }

  // transition relation

  if(!op_trans.is_true())
  {
    message.status("Transition relation");

    for(unsigned c=0; c<map.get_no_timeframes()-1; c++)
    {
      message.status("Transition "+i2string(c)+"->"+i2string(c+1));

      instantiate_constraint(solver, map, op_trans, c, c+1, ns, message);
    }
  }
}

/*******************************************************************\

Function: unwind

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void unwind(
  const transt &trans,
  messaget &message,
  decision_proceduret &decision_procedure,
  unsigned no_timeframes,
  const namespacet &ns,
  bool initial_state)
{
  const exprt &op_invar=trans.invar();
  const exprt &op_init=trans.init();
  const exprt &op_trans=trans.trans();

  // general constraints

  message.status("General constraints");

  if(!op_invar.is_true())
    for(unsigned c=0; c<no_timeframes; c++)
      instantiate(decision_procedure, op_invar, c, ns);

  // initial state

  if(initial_state)
  {
    message.status("Initial state");

    if(!op_init.is_true())
      instantiate(decision_procedure, op_init, 0, ns);
  }

  // transition relation

  message.status("Transition relation");

  if(!op_trans.is_true())
    for(unsigned c=0; c<no_timeframes-1; c++)
    {
      message.status("Transition "+i2string(c)+"->"+i2string(c+1));

      instantiate(decision_procedure, op_trans, c, ns);
    }
}
