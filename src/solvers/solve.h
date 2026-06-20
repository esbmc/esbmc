#ifndef _ESBMC_SOLVERS_SOLVE_H_
#define _ESBMC_SOLVERS_SOLVE_H_

#include <string>
#include <util/config.h>
#include <util/namespace.h>
#include <util/message.h>

class array_iface;
class fp_convt;
class smt_convt;
class smt_solver_baset;
class tuple_iface;

typedef smt_solver_baset *(solver_creator)(
  const optionst &options,
  const namespacet &ns,
  tuple_iface **tuple_api,
  array_iface **array_api,
  fp_convt **fp_api);

smt_convt *create_solver(
  std::string solver_name,
  const namespacet &ns,
  const optionst &options);

/// Abort early if the user explicitly selected an SMT solver that was not
/// built into this ESBMC binary. Safe to call before parsing the program;
/// returns silently when no explicit selection was made (the default picker
/// will choose from whatever is built in at solver-creation time).
void check_solver_availability(const optionst &options);

#endif
