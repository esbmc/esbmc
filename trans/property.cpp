/*******************************************************************\

Module: Unwinding the Properties

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <namespace.h>
#include <cout_message.h>
#include <i2string.h>

#include "instantiate.h"
#include "property.h"

/*******************************************************************\

Function: property

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void property(
  const std::list<exprt> &properties,
  std::list<bvt> &prop_bv,
  messaget &message,
  propt &solver,
  const bmc_mapt &map,
  const namespacet &ns)
{
  if(properties.size()==1)
    message.status("Adding property");
  else
    message.status("Adding "+i2string(properties.size())+" properties");

  prop_bv.clear();
  bvt all_prop;

  for(std::list<exprt>::const_iterator
      it=properties.begin();
      it!=properties.end();
      it++)
  {
    if(it->is_true())
    {
      prop_bv.push_back(bvt());
      prop_bv.back().resize(map.get_no_timeframes(), const_literal(true));
      continue;
    }
  
    exprt property(*it);

    if(property.id()!="AG" ||
       property.operands().size()!=1)
    {
      message.error("unsupported property - only AGp implemented");
      exit(1);
    }

    const exprt &p=property.op0();
    
    prop_bv.push_back(bvt());

    for(unsigned c=0; c<map.get_no_timeframes(); c++)
    {
      literalt l=instantiate_convert(solver, map, p, c, c+1, ns, message);
      prop_bv.back().push_back(l);
      all_prop.push_back(solver.lnot(l));
    }
  }

  solver.lcnf(all_prop);
}

/*******************************************************************\

Function: property

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void property(
  const std::list<exprt> &properties,
  std::list<bvt> &prop_bv,
  messaget &message,
  prop_convt &solver,
  unsigned no_timeframes,
  const namespacet &ns)
{
  if(properties.size()==1)
    message.status("Adding property");
  else
    message.status("Adding "+i2string(properties.size())+" properties");

  prop_bv.clear();
  bvt all_prop;

  for(std::list<exprt>::const_iterator
      it=properties.begin();
      it!=properties.end();
      it++)
  {
    if(it->is_true())
    {
      prop_bv.push_back(bvt());
      prop_bv.back().resize(no_timeframes, const_literal(true));
      continue;
    }
  
    exprt property(*it);

    if(property.id()!="AG" ||
       property.operands().size()!=1)
    {
      message.error("unsupported property - only AGp implemented");
      exit(1);
    }

    const exprt &p=property.op0();
    
    prop_bv.push_back(bvt());

    for(unsigned c=0; c<no_timeframes; c++)
    {
      exprt tmp(p);
      instantiate(tmp, c, ns);

      literalt l=solver.convert(tmp);
      prop_bv.back().push_back(l);
      all_prop.push_back(solver.prop.lnot(l));
    }
  }

  solver.prop.lcnf(all_prop);
}
