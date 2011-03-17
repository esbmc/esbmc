/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_TRANS_UNWIND_H
#define CPROVER_TRANS_UNWIND_H

#include <namespace.h>
#include <message.h>
#include <hash_cont.h>
#include <decision_procedure.h>
#include <std_expr.h>

#include <solvers/prop/prop.h>

#include "bmc_map.h"

// bit-level

void unwind(
  const transt &_trans,
  messaget &message,
  propt &solver,
  bmc_mapt &map,
  const namespacet &ns);

// word-level

void unwind(
  const transt &trans,
  messaget &message,
  decision_proceduret &decision_procedure,
  unsigned no_timeframes,
  const namespacet &ns,
  bool initial_state=true);

#endif
