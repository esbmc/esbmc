/*******************************************************************\

Module: Symbolic Execution

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_BASIC_SYMEX_H
#define CPROVER_BASIC_SYMEX_H

#include <map>
#include <set>

#include <options.h>
#include <namespace.h>
#include <replace_expr.h>
#include <std_code.h>

#include "execution_state.h"
#include "symex_target.h"
#include "goto_symex_state.h"

class basic_symext
{
public:
  basic_symext()
  {
//    options.set_option("no-simplify", false);
  }

  virtual ~basic_symext() { }
};

#endif
