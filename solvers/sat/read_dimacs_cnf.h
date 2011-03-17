/*******************************************************************\

Module: Reading DIMACS CNF

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef __ESBMC_READ_DIMACS_CNF_H
#define __ESBMC_READ_DIMACS_CNF_H

#include "cnf.h"

void read_dimacs_cnf(std::istream &in, cnft &dest);

#endif
