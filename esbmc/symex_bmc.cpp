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
