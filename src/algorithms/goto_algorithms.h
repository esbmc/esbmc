#ifndef ESBMC_GOTO_ALGORITHMS_H
#define ESBMC_GOTO_ALGORITHMS_H

#include <algorithms/algorithm.h>
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
class unwind_goto_functions : public goto_functions_algorithm
{
public:
  unwind_goto_functions(goto_functionst &goto_functions, message_handlert &msg)
    : goto_functions_algorithm(goto_functions), msg(msg)
  {
  }
  bool run() override;

  unsigned get_number_of_functions() { return number_of_functions; }
  unsigned get_number_of_loops() { return number_of_loops; }
  unsigned get_number_of_bounded_loops() { return number_of_bounded_loops; }

protected:
  void unroll_loop(goto_programt &goto_program, loopst &loop);
  unsigned number_of_functions = 0;
  unsigned number_of_loops = 0;
  unsigned number_of_bounded_loops = 0;

private:  
  message_handlert &msg; // This is needed to get the program loop
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
class bounded_unwind_goto_functions : public unwind_goto_functions
{
public:
  bounded_unwind_goto_functions(
    goto_functionst &goto_functions,
    message_handlert &msg)
    : unwind_goto_functions(goto_functions, msg)
  {
    unsupported_options = {"unwind", "context-bound"};
  }
};

/**
 * @brief This algorithm tries to estimate what is the bound of a loop
 * by doing a pattern matching over the loop.
 * 
 * This is just the naive approach, we should research advanced methods
 * for this.
 * 
 */
class get_loop_bounds : public goto_program_algorithm
{
public:
  get_loop_bounds(goto_programt &goto_program, loopst &loop)
    : goto_program_algorithm(goto_program), loop(loop)
  {
  }
  unsigned get_bound()
  {
    return bound;
  }
  bool run() override;

protected:
  loopst &loop;
  unsigned bound = 0;
};
#endif //ESBMC_GOTO_ALGORITHMS_H