/*******************************************************************\

Module: Graph representing Netlist

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_TRANS_NETLIST_H
#define CPROVER_TRANS_NETLIST_H

#include <iostream>

#include <std_expr.h>
#include <namespace.h>

#include <solvers/prop/aig.h>

#include "var_map.h"

class var_map_labelingt:public aig_variable_labelingt
{
protected:
  const var_mapt &var_map;

public:
  var_map_labelingt(const var_mapt &_var_map):var_map(_var_map)
  {
  }

  virtual std::string operator()(unsigned v) const
  {
    return id2string(var_map.reverse_map[v].id);
  }
  
  virtual std::string dot_label(unsigned v) const;
};

class netlistt:public aigt
{
public:
  var_mapt var_map;

  netlistt()
  {
  }

  literalt current(unsigned var_no) const
  {
    assert(var_no<current_state.size());
    return current_state[var_no];
  }
  
  literalt next(unsigned var_no) const
  {
    assert(var_no<next_state.size());
    return next_state[var_no];
  }
  
  unsigned get_no_vars() const
  {
    return var_map.get_no_vars();
  }
  
  void swap(netlistt &other)
  {
    aigt::swap(other);
    other.var_map.swap(var_map);
    other.current_state.swap(current_state);
    other.next_state.swap(next_state);
  }
  
  virtual void print(std::ostream &out) const;
  virtual void print(std::ostream &out, literalt a) const;
  virtual void output_dot(std::ostream &out) const;
  
  void set_current(unsigned var_no, literalt l)
  {
    current_state.resize(get_no_vars());
    assert(var_no<get_no_vars());
    current_state[var_no]=l;
  }

  void set_next(unsigned var_no, literalt l)
  {
    next_state.resize(get_no_vars());
    assert(var_no<get_no_vars());
    next_state[var_no]=l;
  }

protected:
  // map from variable to node number
  typedef std::vector<literalt> var_node_mapt;

  // the 'current_state' literals are always positive
  // no guarantee like that for the next_state literals
  var_node_mapt current_state, next_state;  
};

#endif
