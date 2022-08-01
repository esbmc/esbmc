#ifndef ESBMC_ALGORITHM_H
#define ESBMC_ALGORITHM_H

#include <goto-programs/goto_functions.h>
#include <goto-programs/loopst.h>
#include <goto-programs/goto_loops.h>
#include <util/message.h>
/**
 * @brief Base interface to run an algorithm in esbmc
 */
template <typename T>
class algorithm
{
public:
  algorithm(bool sideeffect) : sideeffect(sideeffect)
  {
  }

  /**
   * @brief Executes the algorithm over a T object
   *
   * @return success of the algorithm
   */
  virtual bool run(T &) = 0;

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

protected:
  virtual bool runOnFunction(std::pair<const dstring, goto_functiont> &F);
  virtual bool runOnLoop(loopst &loop, goto_programt &goto_program);

private:
  unsigned number_of_functions = 0;
  unsigned number_of_loops = 0;
};

#endif //ESBMC_ALGORITHM_H
