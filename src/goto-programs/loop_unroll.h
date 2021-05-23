#ifndef ESBMC_GOTO_ALGORITHMS_H
#define ESBMC_GOTO_ALGORITHMS_H

#include <util/algorithms.h>
#include <util/message.h>

/**
 * @brief This is the base class that unroll
 * an loop (in an unsound way), the idea is to
 * copy-paste the contents of the loop N times
 * and ignore its conditions.
 * 
 * If you need a sound approach for this unrolling
 * you can try commit: d4a196881af560568960d8eb48b5b22ef4bddf5e
 * 
 */
class unsound_loop_unroller : public goto_functions_algorithm
{
public:
  unsound_loop_unroller(
    goto_functionst &goto_functions,
    message_handlert &msg,
    std::string disable_cmd)
    : goto_functions_algorithm(goto_functions, msg, disable_cmd, true)
  {
  }

protected:
  bool runOnLoop(loopst &loop, goto_programt &goto_program) override;

  /**
   * @brief Get the number of iteractions to unroll the loop
   * 
   * @param loop 
   * @return int negative means error, positive is the quantity to unroll
   */
  virtual int get_loop_bounds(loopst &loop) = 0;
};

/**
 * @brief This goes through every loop of the program and tries to
 * check if it is a bounded loop, if it is, it will be unrolled all
 * times needed. The intentation is to simplify structures such as:
 * 
 * SYMBOL = K0
 * 1: IF !(SYMBOL < K) THEN GOTO 2
 *    P
 *    SYMBOL = SYMBOL + 1
 *    GOTO 1
 * 2: Q
 * 
 * If K > K0 and P does not contain assignments over SYMBOL, then this
 * is converted into:
 * 
 * P
 * SYMBOL = SYMBOL + 1
 * ...  // K - K0 times
 * P 
 * SYMBOL = SYMBOL + 1
 * Q
 */
class bounded_loop_unroller : public unsound_loop_unroller
{
public:
  bounded_loop_unroller(goto_functionst &goto_functions, message_handlert &msg)
    : unsound_loop_unroller(goto_functions, msg, "no-unroll")
  {
    unsupported_options = {"unwind", "context-bound"};
  }

  unsigned get_number_of_bounded_loops()
  {
    return number_of_bounded_loops;
  }

protected:
  int get_loop_bounds(loopst &loop) override;

private:
  unsigned number_of_bounded_loops = 0;
};

#endif