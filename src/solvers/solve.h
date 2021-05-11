#ifndef _ESBMC_SOLVERS_SOLVE_H_
#define _ESBMC_SOLVERS_SOLVE_H_

#include <solvers/smt_conv.h>
#include <util/config.h>
#include <util/namespace.h>

smt_convt *create_solver_factory(const namespacet &ns, const optionst &options);

#endif
