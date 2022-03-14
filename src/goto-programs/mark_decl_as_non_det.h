#pragma once

#include <util/algorithms.h>
#include <irep2/irep2.h>
#include <util/message/message.h>

/**
 * @brief This will look over non-initialized local declarations
 * and set them as nondet
 *
 */
class mark_decl_as_non_det : public goto_functions_algorithm
{
public:
  mark_decl_as_non_det(contextt &context, goto_functionst &goto_functions)
    : goto_functions_algorithm(goto_functions, true), context(context)
  {
  }

protected:
  contextt &context;
  virtual bool
  runOnFunction(std::pair<const dstring, goto_functiont> &F) override;
};