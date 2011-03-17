/*******************************************************************\

Module: Variable Mapping

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <assert.h>

#include "map_aigs.h"

/*******************************************************************\

Function: map_to_timeframe

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void map_to_timeframe(const bmc_mapt &bmc_map, aigt &aig, unsigned t)
{
  assert(t<bmc_map.timeframe_map.size());
  const bvt &timeframe=bmc_map.timeframe_map[t];

  for(unsigned i=0; i<aig.nodes.size(); i++)
  {
    aig_nodet &node=aig.nodes[i];

    if(node.is_var())
    {
      unsigned old=node.var_no();
      assert(old<timeframe.size());
      node.make_var(timeframe[old].var_no());
    }
  }
}

/*******************************************************************\

Function: map_from_timeframe

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void map_from_timeframe(const bmc_mapt &bmc_map, aigt &aig, unsigned t)
{
  // need to build reverse-map first
  typedef std::map<unsigned, unsigned> reverse_mapt;
  reverse_mapt reverse_map;

  assert(t<bmc_map.timeframe_map.size());
  const bvt &timeframe=bmc_map.timeframe_map[t];

  for(unsigned i=0; i<timeframe.size(); i++)
  {
    literalt l=timeframe[i];
    assert(!l.sign());
    reverse_map[l.var_no()]=i;
  }

  // now apply
  for(unsigned i=0; i<aig.nodes.size(); i++)
  {
    aig_nodet &node=aig.nodes[i];

    if(node.is_var())
    {
      unsigned old=node.var_no();
      const reverse_mapt::const_iterator it=reverse_map.find(old);
      assert(it!=reverse_map.end());
      node.make_var(it->second);
    }
  }
}

