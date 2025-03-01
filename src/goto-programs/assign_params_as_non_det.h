#pragma once

#include <util/algorithms.h>
#include <util/expr_util.h>
#include <irep2/irep2.h>
#include <util/message.h>
#include <util/arith_tools.h>

/**
 * @brief To make the counterexample trace complete in "--function" mode
 *
 */
class assign_params_as_non_det : public goto_functions_algorithm
{
public:
  assign_params_as_non_det(contextt &context)
    : goto_functions_algorithm(true), context(context)
  {
  }

protected:
  contextt &context;
  virtual bool
  runOnFunction(std::pair<const dstring, goto_functiont> &F) override;
  symbolt *get_default_symbol(
    typet type,
    std::string name,
    std::string id,
    locationt location);
  bool assign_nondet(
    const exprt &arg,
    goto_programt &goto_program,
    goto_programt::instructiont::targett &it,
    locationt l);
};
