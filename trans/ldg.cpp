/*******************************************************************\

Module: Latch Dependency Graph

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <iostream>

#include <assert.h>

#include "ldg.h"

/*******************************************************************\

Function: ldgt::compute

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void ldgt::compute(const netlistt &netlist)
{
  compute(netlist, netlist.var_map.latches);
}

/*******************************************************************\

Function: ldgt::compute

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void ldgt::compute(
  const netlistt &netlist,
  const latchest &localization)
{
  latches=netlist.var_map.latches;
  
  // build a reverse mapping for the outputs
  // of the latches
  node_infost node_infos;
  
  node_infos.resize(netlist.number_of_nodes());
  
  for(latchest::const_iterator
      l_it=localization.begin();
      l_it!=localization.end();
      l_it++)
  {
    unsigned n=netlist.current(*l_it).var_no();
    node_infos[n].is_source_latch=true;
    node_infos[n].var_no=*l_it;
  }
  
  // we start with a node for each variable
  nodes.clear();
  nodes.resize(netlist.var_map.get_no_vars());
  
  aigt::terminalst terminals;
  netlist.get_terminals(terminals);

  for(latchest::const_iterator
      l_it=localization.begin();
      l_it!=localization.end();
      l_it++)
  {
    unsigned v=*l_it;
    literalt l=netlist.next(v);
    const aigt::terminal_sett &t=terminals[l.var_no()];
    
    for(std::set<unsigned>::const_iterator
        it=t.begin(); it!=t.end(); it++)
    {
      if(node_infos[*it].is_source_latch)
      {
        unsigned v2=node_infos[*it].var_no;
        add_edge(v2, v);
        #if 0
        std::cout << "DEP: " << v2 << " -> " << v << std::endl;
        #endif
      }
    }
  }
}
