#pragma once

#include <goto-programs/goto_functions_algorithm.h>
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
  mark_decl_as_non_det(contextt &context, bool show_uninitialized)
    : goto_functions_algorithm(true),
      context(context),
      show_uninitialized(show_uninitialized)
  {
  }

protected:
  contextt &context;
  const bool show_uninitialized;
  virtual bool
  runOnFunction(std::pair<const irep_idt, goto_functiont> &F) override;
};
