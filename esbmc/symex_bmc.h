/*******************************************************************\

Module: Bounded Model Checking for ANSI-C

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_CBMC_SYMEX_BMC_H
#define CPROVER_CBMC_SYMEX_BMC_H

#include <hash_cont.h>
#include <message.h>
#include <goto-symex/reachability_tree.h>

class symex_bmct:
  public reachability_treet,
  virtual public messaget
{
public:
  symex_bmct(
    const goto_functionst &goto_functions,
    optionst &opts,
    const namespacet &_ns,
    contextt &_new_context,
    symex_targett &_target);

  friend class bmct;

  ~symex_bmct(){ };

protected:
  //
  // overloaded from goto_symext
  //
};

#endif
