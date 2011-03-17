/*******************************************************************\

Module: Variable Mapping

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_TRANS_BMC_MAP_H
#define CPROVER_TRANS_BMC_MAP_H

#include <assert.h>

#include "var_map.h"

typedef std::vector<bvt> timeframe_mapt;

class bmc_mapt
{
public:
  literalt get(unsigned timeframe, const var_mapt::vart::bitt &bit) const
  {
    return get(timeframe, bit.var_no);
  }

  literalt get(unsigned timeframe, const irep_idt &id, unsigned bit_nr) const
  {
    return get(timeframe, var_map.get(id, bit_nr));
  }

  literalt get(unsigned timeframe, unsigned var_no) const
  {
    assert(timeframe<timeframe_map.size());
    assert(var_no<timeframe_map[timeframe].size());
    return timeframe_map[timeframe][var_no];
  }

  void set(unsigned timeframe, unsigned var_no, literalt l)
  {
    assert(timeframe<timeframe_map.size());
    assert(var_no<timeframe_map[timeframe].size());
    timeframe_map[timeframe][var_no]=l;
  }

  literalt get(unsigned timeframe, const bv_varidt &varid) const
  {
    return get(timeframe, var_map.get(varid));
  }

  void map_vars(
    const contextt &context,
    const irep_idt &module)
  {
    var_map.map_vars(context, module);
  }

  // number of valid timeframes
  // this is number of cycles +1!
  void map_timeframes(propt &solver, unsigned no_timeframes);
  
  // get the timeframe map
  void map_timeframes_latches(timeframe_mapt &map);

  var_mapt var_map;
  timeframe_mapt timeframe_map;
  
  void get_latch_vector(
    unsigned timeframe,
    propt &solver,
    std::vector<bool> &values) const;
  
  void get_latch_literals(
    unsigned timeframe,
    bvt &dest) const;
  
  unsigned get_no_timeframes() const
  {
    return timeframe_map.size();
  }
   
  bmc_mapt() { }
  
  virtual ~bmc_mapt()
  {
  }
  
  void clear()
  {
    timeframe_map.clear();
    var_map.clear();
  }
};

#endif
