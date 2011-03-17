/*******************************************************************\

Module: Unwinding Netlists

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <i2string.h>

#include "unwind_netlist.h"

/*******************************************************************\

Function: netlist_bmc_mapt::build_bmc_map

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void netlist_bmc_mapt::build_bmc_map(
  const netlistt &netlist,
  bmc_mapt &bmc_map) const
{
  bmc_map.var_map=netlist.var_map;
  bmc_map.timeframe_map.resize(timeframe_map.size());

  for(unsigned t=0; t<timeframe_map.size(); t++)
  {
    const timeframet &timeframe=timeframe_map[t];
    bvt &bmc_map_bv=bmc_map.timeframe_map[t];
    
    bmc_map_bv.resize(bmc_map.var_map.get_no_vars());

    for(unsigned v=0; v<bmc_map_bv.size(); v++)
    {
      literalt l;

      if(netlist.current(v).is_true())
        l=const_literal(true);
      else if(netlist.current(v).is_false())
        l=const_literal(false);
      else
      {
        unsigned node=netlist.current(v).var_no();
        assert(node<timeframe.nodes.size());
        l=timeframe.nodes[node].l;
      }

      bmc_map_bv[v]=l;
    }
  }
}

/*******************************************************************\

Function: netlist_bmc_mapt::map_timeframes

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void netlist_bmc_mapt::map_timeframes(
  const netlistt &netlist,
  unsigned no_timeframes,
  propt &solver)
{
  timeframe_map.resize(no_timeframes);
  
  for(unsigned t=0; t<timeframe_map.size(); t++)
  {
    timeframet &timeframe=timeframe_map[t];
    timeframe.nodes.resize(netlist.number_of_nodes());

    for(unsigned n=0; n<timeframe.nodes.size(); n++)
      timeframe.nodes[n].l=solver.new_variable();
  }
}

/*******************************************************************\

Function: unwind

  Inputs:

 Outputs:

 Purpose: Unwind timeframe by timeframe

\*******************************************************************/

void unwind(
  const netlist_transt &netlist,
  netlist_bmc_mapt &netlist_bmc_map,
  messaget &message,
  cnft &solver,
  bool add_initial_state,
  unsigned t)
{
  if(add_initial_state && t==0)
  {
    // do initial state
    message.status("Initial State");
    solver.l_set_to(
      netlist_bmc_map.translate(0, netlist.initial),
      true);
  }

  // do transitions
  bool last=(t==netlist_bmc_map.timeframe_map.size()-1);

  if(last)
    message.status("Transition "+i2string(t));
  else
    message.status("Transition "+i2string(t)+"->"+i2string(t+1));
  
  const netlist_bmc_mapt::timeframet &timeframe=netlist_bmc_map.timeframe_map[t];
  
  for(unsigned n=0; n<timeframe.nodes.size(); n++)
  {
    const aig_nodet &node=netlist.get_node(n);

    if(node.is_and() && timeframe.nodes[n].is_visible)
    {
      literalt la=netlist_bmc_map.translate(t, node.a);
      literalt lb=netlist_bmc_map.translate(t, node.b);
    
      solver.land(la, lb, timeframe.nodes[n].l);
    }
  }

  // transition constraint
  solver.l_set_to(
    netlist_bmc_map.translate(t, netlist.transition),
    true);

  if(!last)
  {     
    // joining timeframe and timeframe+1
    for(unsigned v=0; v<netlist.get_no_vars(); v++)
    {
      literalt l_from=netlist.next(v);
      literalt l_to=netlist.current(v);

      if(l_from.is_constant() ||
         netlist_bmc_map.timeframe_map[t].nodes[l_from.var_no()].is_visible)
        solver.set_equal(
          netlist_bmc_map.translate(t, l_from),
          netlist_bmc_map.translate(t+1, l_to));
    }
  }
}

/*******************************************************************\

Function: unwind

  Inputs:

 Outputs:

 Purpose: 

\*******************************************************************/

void unwind(
  const netlist_transt &netlist,
  netlist_bmc_mapt &netlist_bmc_map,
  messaget &message,
  cnft &solver,
  bool add_initial_state)
{
  for(unsigned t=0; t<netlist_bmc_map.timeframe_map.size(); t++)
    unwind(netlist, netlist_bmc_map, message, solver, add_initial_state, t);
}

/*******************************************************************\

Function: unwind_property

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void unwind_property(
  const netlist_transt &netlist,
  const netlist_bmc_mapt &netlist_bmc_map,
  messaget &message,
  std::list<bvt> &prop_bv,
  cnft &solver)  
{
  if(netlist.properties.empty()) return;

  message.status("Unwinding property");
  
  bvt or_bv;
  
  or_bv.reserve(
    netlist.properties.size()*netlist_bmc_map.timeframe_map.size());

  for(unsigned p=0; p<netlist.properties.size(); p++)
  {
    prop_bv.push_back(bvt());
    prop_bv.back().reserve(netlist_bmc_map.timeframe_map.size());

    for(unsigned t=0;
        t<netlist_bmc_map.timeframe_map.size();
        t++)
    {
      literalt l=netlist_bmc_map.translate(t, netlist.properties[p]);
      or_bv.push_back(solver.lnot(l));
      prop_bv.back().push_back(l);
    }

    assert(prop_bv.back().size()==netlist_bmc_map.timeframe_map.size());
  }
  
  assert(prop_bv.size()==netlist.properties.size());  
    
  solver.lcnf(or_bv);
}
