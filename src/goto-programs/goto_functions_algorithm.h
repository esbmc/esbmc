#pragma once

#include <goto-programs/goto_functions.h>
#include <goto-programs/loopst.h>
#include <util/base/algorithm.h>

/**
 * @brief Base interface for goto-functions algorithms
 */
class goto_functions_algorithm : public algorithm<goto_functionst>
{
public:
  explicit goto_functions_algorithm(bool sideffect) : algorithm(sideffect)
  {
  }

  unsigned get_number_of_functions()
  {
    return number_of_functions;
  }
  unsigned get_number_of_loops()
  {
    return number_of_loops;
  }

  bool run(goto_functionst &) override;
  void setTarget(const std::string &_tgt)
  {
    target_function = _tgt;
  }

protected:
  virtual bool runOnFunction(std::pair<const irep_idt, goto_functiont> &F);
  virtual bool runOnLoop(loopst &loop, goto_programt &goto_program);
  virtual bool runOnProgram(goto_functionst &)
  {
    return false;
  }
  std::string target_function = "";

private:
  unsigned number_of_functions = 0;
  unsigned number_of_loops = 0;
};
