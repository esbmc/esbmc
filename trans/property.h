/*******************************************************************\

Module: Unwinding the Properties

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_TRANS_PROPERTY_H
#define CPROVER_TRANS_PROPERTY_H

#include <context.h>
#include <message.h>
#include <namespace.h>

#include <solvers/prop/prop.h>
#include <solvers/prop/prop_conv.h>

#include "bmc_map.h"

// bit-level
void property(
  const std::list<exprt> &properties,
  std::list<bvt> &prop_bv,
  messaget &message,
  propt &solver,
  const bmc_mapt &map,
  const namespacet &ns);

// word-level
void property(
  const std::list<exprt> &properties,
  std::list<bvt> &prop_bv,
  messaget &message,
  prop_convt &solver,
  unsigned no_timeframes,
  const namespacet &ns);

#endif
