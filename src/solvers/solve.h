#ifndef _ESBMC_SOLVERS_SOLVE_H_
#define _ESBMC_SOLVERS_SOLVE_H_

#include <solvers/smt/smt_conv.h>
#include <string>
#include <util/config.h>
#include <util/namespace.h>
#include <util/message.h>

typedef smt_convt *(solver_creator)(
  const optionst &options,
  const namespacet &ns,
  tuple_iface **tuple_api,
  array_iface **array_api,
  fp_convt **fp_api);

smt_convt *create_solver(
  std::string solver_name,
  const namespacet &ns,
  const optionst &options);

#endif
