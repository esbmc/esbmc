#ifndef _ESBMC_SOLVERS_SOLVE_H_
#define _ESBMC_SOLVERS_SOLVE_H_

#include <string>

#include <config.h>
#include <namespace.h>

#include <solvers/smt/smt_conv.h>

smt_convt *create_solver_factory(const std::string &solver_name,
                                  bool int_encoding, const namespacet &ns,
                                  const optionst &options);

#endif
