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
  basic_symext(
    const namespacet &_ns,
    contextt &_new_context,
    symex_targett &_target):
    constant_propagation(true),
    new_context(_new_context),
    ns(_ns),
    target(&_target)
  {
    options.set_option("no-simplify", false);
  }

  virtual ~basic_symext() { }

  typedef goto_symex_statet statet;

  bool constant_propagation;

  optionst options;
  contextt &new_context;

// XXX jmorse - un-protected to appease bmc.cpp. g++34 does not approve of its
// access of this member.
  const namespacet &ns;
  symex_targett *target;

protected:
  virtual void do_simplify(exprt &expr);

  virtual void replace_dynamic_allocation(const statet &state, exprt &expr);
  void replace_nondet(exprt &expr, execution_statet &ex_state);
};

#endif
