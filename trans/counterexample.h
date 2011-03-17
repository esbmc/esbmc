/*******************************************************************\

Module: Extracting Counterexamples

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_BMC_COUNTEREXAMPLE_H
#define CPROVER_BMC_COUTNEREXAMPLE_H

#include <namespace.h>
#include <message.h>
#include <solvers/prop/prop.h>
#include <langapi/language_ui.h>

#include "bmc_map.h"

void show_counterexample(
  const std::list<exprt> &properties,
  const std::list<bvt> &prop_bv,
  messaget &message,
  const propt &solver,
  const bmc_mapt &map,
  const namespacet &ns,
  language_uit::uit ui);

void show_counterexample(
  messaget &message,
  const propt &solver,
  const bmc_mapt &map,
  const namespacet &ns,
  language_uit::uit ui);

void show_counterexample(
  messaget &message,
  const class decision_proceduret &solver,
  unsigned no_timeframes,
  const namespacet &ns,
  const std::string &module,
  language_uit::uit ui);

#endif
