/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_TRANS_UNWIND_NETLIST_GRAPH_H
#define CPROVER_TRANS_UNWIND_NETLIST_GRAPH_H

#include <assert.h>

#include <message.h>
#include <solvers/sat/cnf.h>

#include "netlist_trans.h"
#include "bmc_map.h"

class netlist_bmc_mapt
{
public:
  struct timeframet
  {
  public:
    struct nodet
    {
      literalt l;
      bool is_visible;
      
      nodet():is_visible(true)
      {
      }
    };
    
    typedef std::vector<nodet> nodest;
    nodest nodes;
  };

  std::vector<timeframet> timeframe_map;
  
  void build_bmc_map(
    const netlistt &netlist,
    bmc_mapt &bmc_map) const;  

  // number of valid timeframes
  // this is number of cycles +1!
  void map_timeframes(
    const netlistt &netlist,
    unsigned no_timeframes,
    propt &solver);

  // translate and-graph literal into propositional literal    
  literalt translate(unsigned timeframe, literalt l) const
  {
    if(l.is_constant()) return l;
    assert(timeframe<timeframe_map.size());
    return timeframe_map[timeframe].nodes[l.var_no()].l.cond_negation(l.sign());
  }
  
  void translate(unsigned timeframe, bvt &bv) const
  {
    for(unsigned i=0; i<bv.size(); i++)
      bv[i]=translate(timeframe, bv[i]);
  }
  
  void set_visible(unsigned timeframe, literalt l, bool is_visible)
  {
    if(l.is_constant()) return;
    assert(timeframe<timeframe_map.size());
    timeframe_map[timeframe].nodes[l.var_no()].is_visible=is_visible;
  }
  
  void make_invisible(unsigned timeframe, literalt l)
  {
    set_visible(timeframe, l, false);
  }
  
  void make_visible(unsigned timeframe, literalt l)
  {
    set_visible(timeframe, l, true);
  }
  
  void make_all_invisible()
  {
    for(unsigned t=0; t<timeframe_map.size(); t++)
      for(unsigned n=0; n<timeframe_map[t].nodes.size(); n++)
        timeframe_map[t].nodes[n].is_visible=false;
  }
};

void unwind(
  const netlist_transt &netlist,
  netlist_bmc_mapt &netlist_bmc_map,
  messaget &message,
  cnft &solver,
  bool add_initial_state=true);

void unwind(
  const netlist_transt &netlist_trans,
  netlist_bmc_mapt &netlist_bmc_map,
  messaget &message,
  cnft &solver,
  bool add_initial_state,
  unsigned timeframe);

void unwind_property(
  const netlist_transt &netlist,
  const netlist_bmc_mapt &netlist_bmc_map,
  messaget &message,
  std::list<bvt> &prop_bv,
  cnft &solver);

#endif
