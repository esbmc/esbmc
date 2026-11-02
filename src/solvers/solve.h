#ifndef _ESBMC_SOLVERS_SOLVE_H_
#define _ESBMC_SOLVERS_SOLVE_H_

#include <string>
#include <util/config/config.h>
#include <util/symtab/namespace.h>
#include <util/message/message.h>

class array_iface;
class fp_convt;
class smt_convt;
class smt_solver_baset;
class tuple_iface;

typedef smt_solver_baset *(solver_creator)(const optionst &options,
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

/// The solver create_solver("", ...) will use for these options, without
/// logging the choice; "" if no backend can be picked implicitly.
std::string effective_solver_name(const optionst &options);

/// Whether \p solver_name's native floating-point theory has the format with
/// \p exponent exponent bits and \p fraction fraction bits (hidden bit
/// excluded). Solvers without such a restriction answer true.
bool solver_supports_fp_format(
  const std::string &solver_name,
  unsigned exponent,
  unsigned fraction);

#endif
