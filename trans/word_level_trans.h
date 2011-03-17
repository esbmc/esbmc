/*******************************************************************\

Module: Word-level transition relation

Author: Daniel Kroening

Date: December 2005

Purpose:

\*******************************************************************/

#ifndef CPROVER_WORD_LEVEL_TRANS_H
#define CPROVER_WORD_LEVEL_TRANS_H

#include <hash_cont.h>
#include <std_expr.h>
#include <namespace.h>

class word_level_transt
{
public:
  word_level_transt(const namespacet &_ns):
    ns(_ns)
  {
  }

  // read transition system into next state function
  void read_trans(const transt &trans);

  void clear()
  {
    next_state_functions.clear();
  }
  
  void output(std::ostream &out);

  typedef hash_map_cont<irep_idt, exprt, irep_id_hash> next_state_functionst;
  next_state_functionst next_state_functions;

protected:
  const namespacet &ns;

  void read_trans_rec(const exprt &expr);
  void equality(const exprt &lhs, const exprt &rhs);
};

#endif
