/*******************************************************************\
 Module: Algorithm Interface
 Author: Rafael Sá Menezes
 Date: May 2021

 Description: The algorithm interface is to be used for
              every kind of logic that uses a generic datastructure
              to be reasoned: containers, CFG, goto-programs, loops.

              The idea is that we don't need to look over a million
              of lines in the flow of esbmc when we only want to do 
              a small analysis.
\*******************************************************************/

#ifndef ESBMC_ALGORITHM_H
#define ESBMC_ALGORITHM_H

#include <goto-programs/goto_functions.h>
#include <goto-programs/loopst.h>
#include <goto-programs/goto_loops.h>
#include <util/message/message.h>
/**
 * @brief Base interface to run an algorithm in esbmc
 */
class algorithm
{
public:
  algorithm(bool sideeffect) : sideeffect(sideeffect)
  {
  }

  /**
   * @brief Executes the algorithm
   * 
   * @return success of the algorithm
   */
  virtual bool run() = 0;

  /**
   * @brief Says wether the algorithm is a plain analysis
   * or if it also changes the structure
   * 
   */
  bool has_sideeffect()
  {
    return sideeffect;
  }

protected:
  // The algorithm changes the CFG, container in some way?
  const bool sideeffect;
};

/**
 * @brief Base interface for goto-functions algorithms
 */
class goto_functions_algorithm : public algorithm
{
public:
  explicit goto_functions_algorithm(
    goto_functionst &goto_functions,
    bool sideffect)
    : algorithm(sideffect), goto_functions(goto_functions)
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

  bool run() override;

protected:
  virtual bool runOnFunction(std::pair<const dstring, goto_functiont> &F);
  virtual bool runOnLoop(loopst &loop, goto_programt &goto_program);
  goto_functionst &goto_functions;
  ;

private:
  unsigned number_of_functions = 0;
  unsigned number_of_loops = 0;
};

#endif //ESBMC_ALGORITHM_H