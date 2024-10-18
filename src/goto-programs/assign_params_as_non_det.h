#pragma once

#include <util/algorithms.h>
#include <util/expr_util.h>
#include <irep2/irep2.h>
#include <util/message.h>

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
};
