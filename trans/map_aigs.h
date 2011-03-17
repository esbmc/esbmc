/*******************************************************************\

Module: Variable Mapping

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_TRANS_MAP_AIGS_H
#define CPROVER_TRANS_MAP_AIGS_H

#include <solvers/prop/aig.h>

#include "bmc_map.h"

void map_to_timeframe(const bmc_mapt &bmc_map, aigt &aig, unsigned t);
void map_from_timeframe(const bmc_mapt &bmc_map, aigt &aig, unsigned t);

#endif
