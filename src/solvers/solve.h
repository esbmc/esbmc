#ifndef _ESBMC_SOLVERS_SOLVE_H_
#define _ESBMC_SOLVERS_SOLVE_H_

#include <solvers/smt/smt_conv.h>
#include <string>
#include <util/config.h>
#include <util/namespace.h>

typedef smt_convt *(solver_creator)
    (bool int_encoding, const namespacet &ns, const optionst &opts,
        tuple_iface **tuple_api, array_iface **array_api, fp_convt **fp_api);

typedef smt_convt *(*solver_creator_ptr)
    (bool int_encoding, const namespacet &ns, const optionst &opts,
        tuple_iface **tuple_api, array_iface **array_api, fp_convt **fp_api);

struct esbmc_solver_config {
  std::string name;
  solver_creator_ptr create;
};

extern const struct esbmc_solver_config esbmc_solvers[];
extern const unsigned int esbmc_num_solvers;

smt_convt *create_solver_factory(const std::string &solver_name,
                                  bool int_encoding, const namespacet &ns,
                                  const optionst &options);

#endif
