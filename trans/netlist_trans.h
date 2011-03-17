/*******************************************************************\

Module: Transition System represented by a Netlist

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_TRANS_NETLIST_TRANS_H
#define CPROVER_TRANS_NETLIST_TRANS_H

#include "netlist.h"

class netlist_transt:public netlistt
{
public:
  netlist_transt()
  {
    initial.make_true();
    transition.make_true();
  }

  using netlistt::print;
  virtual void print(std::ostream &out) const;
  virtual void output_dot(std::ostream &out) const;
  
  void swap(netlist_transt &other)
  {
    netlistt::swap(other);
    initial.swap(other.initial);
    transition.swap(other.transition);
  }
  
  // additional constraints
  literalt initial;
  literalt transition;
  
  // properties
  bvt properties;
};

void convert_trans_to_netlist(
  const contextt &context,
  const irep_idt &module,
  const std::list<exprt> &properties,
  class netlist_transt &dest,
  messaget &message);

void convert_trans_to_netlist(
  const namespacet &ns,
  const transt &trans,
  const std::list<exprt> &properties,
  class netlist_transt &dest,
  messaget &message);

#endif
