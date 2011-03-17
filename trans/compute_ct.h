/*******************************************************************\

Module: Completeness Thresholds from LDGs

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_TRANS_COMPUTE_CT_H
#define CPROVER_TRANS_COMPUTE_CT_H

#include "ldg.h"

#define MAX_CT 1000

// returns a CT for a given LDG
unsigned compute_ct(const ldgt &ldg);

#endif
