/*******************************************************************\

Module: Latch Dependency Graph

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_TRANS_LDG_H
#define CPROVER_TRANS_LDG_H

#include <set>

#include <graph.h>

#include <trans/netlist.h>

struct ldg_nodet:public graph_nodet<>
{
public:
};

class ldgt:public graph<ldg_nodet>
{
public:
  typedef std::set<unsigned> latchest;
  latchest latches;

  void compute(const netlistt &netlist);

  // compute with respect to a localization reduction
  // the set contains the variable numbers of the
  // latches that are to be considered
  void compute(const netlistt &netlist,
               const latchest &localization);
  
protected:
  struct node_infot
  {
    bool is_source_latch;
    unsigned var_no;
    
    node_infot():is_source_latch(false)
    {
    }
  };
  
  typedef std::vector<node_infot> node_infost;
};

#endif
