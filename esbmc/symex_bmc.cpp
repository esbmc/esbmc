/*******************************************************************\

Module: Bounded Model Checking for ANSI-C

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <location.h>
#include <i2string.h>

#include "symex_bmc.h"

/*******************************************************************\

Function: symex_bmct::symex_bmct

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

symex_bmct::symex_bmct(
  const goto_functionst &goto_functions,
  optionst &opts,
  const namespacet &_ns,
  contextt &_new_context,
  symex_targett &_target):
  reachability_treet(goto_functions, _ns, opts, _new_context, _target)
{
}

/*******************************************************************\

Function: symex_bmct::get_unwind

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool symex_bmct::get_unwind(
  const symex_targett::sourcet &source,
  unsigned unwind)
{
  unsigned id=source.pc->loop_number;
  unsigned long this_loop_max_unwind=max_unwind;

  if(unwind_set.count(id)!=0)
    this_loop_max_unwind=unwind_set[id];

  #if 1
  {
    std::string msg=
      "Unwinding loop "+i2string(id)+" iteration "+i2string(unwind)+
      " "+source.pc->location.as_string();
    print(8, msg);
  }
  #endif

  return this_loop_max_unwind!=0 &&
         unwind>=this_loop_max_unwind;
}

/*******************************************************************\

Function: symex_bmct::get_unwind_recursion

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool symex_bmct::get_unwind_recursion(
  const irep_idt &identifier,
  unsigned unwind)
{
  unsigned long this_loop_max_unwind=max_unwind;

  #if 1
  if(unwind!=0)
  {
    const symbolt &symbol=ns.lookup(identifier);

    std::string msg=
      "Unwinding recursion "+
      id2string(symbol.display_name())+
      " iteration "+i2string(unwind);

    if(this_loop_max_unwind!=0)
      msg+=" ("+i2string(this_loop_max_unwind)+" max)";

    print(8, msg);
  }
  #endif

  return this_loop_max_unwind!=0 &&
         unwind>=this_loop_max_unwind;
}
