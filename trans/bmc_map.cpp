/*******************************************************************\

Module: 

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <assert.h>

#include <solvers/flattening/boolbv_width.h>

#include "bmc_map.h"

/*******************************************************************\

Function: bmc_mapt::map_timeframes

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void bmc_mapt::map_timeframes(propt &solver, unsigned no_timeframes)
{
  timeframe_map.resize(no_timeframes);

  for(unsigned c=0; c<no_timeframes; c++)
  {
    bvt &bv=timeframe_map[c];

    bv.resize(var_map.get_no_vars());

    for(unsigned i=0; i<bv.size(); i++)
      bv[i]=solver.new_variable();
  }
}

/*******************************************************************\

Function: bmc_mapt::map_timeframes_latches

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void bmc_mapt::map_timeframes_latches(timeframe_mapt &map)
{
  map.resize(timeframe_map.size());

  for(unsigned c=0; c<timeframe_map.size(); c++)
  {
    bvt &bv=map[c];
    bvt &all_bv=timeframe_map[c];

    bv.resize(var_map.latches.size());

    unsigned j=0;

    for(unsigned i=0; i<var_map.get_no_vars(); i++)
      if(var_map.latches.find(i)!=var_map.latches.end())
        bv[j++]=all_bv[i];

    assert(j==bv.size());
  }
}

/*******************************************************************\

Function: bmc_mapt::get_latch_vector

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void bmc_mapt::get_latch_vector(
  unsigned timeframe,
  propt &solver,
  std::vector<bool> &values) const
{
  bvt latch_literals;
  
  get_latch_literals(timeframe, latch_literals);
  
  values.resize(latch_literals.size());
  
  for(unsigned i=0; i<latch_literals.size(); i++)
  {
    switch(solver.l_get(latch_literals[i]).get_value())
    {
    case tvt::TV_TRUE:
      values[i]=true;
      break;
    
    case tvt::TV_FALSE:
      values[i]=false;
      break;
    
    default:
      assert(false);
    }
  }
}

/*******************************************************************\

Function: bmc_mapt::get_latch_literals

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void bmc_mapt::get_latch_literals(
  unsigned timeframe,
  bvt &dest) const
{
  assert(timeframe<timeframe_map.size());

  const bvt &timeframe_vector=timeframe_map[timeframe];
  
  dest.resize(var_map.latches.size());
  
  unsigned i=0;
  
  for(var_mapt::var_sett::const_iterator
      it=var_map.latches.begin();
      it!=var_map.latches.end();
      it++, i++)
    dest[i]=timeframe_vector[*it];
}
