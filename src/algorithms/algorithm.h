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

/**
 * @brief Base interface to run an algorithm in esbmc
 */
class algorithm
{
public:
  algorithm()
  {
  }

  virtual ~algorithm()
  {
  }

  /**
   * @brief Executes the algorithm
   * 
   * @return success of the algorithm
   */
  virtual bool run() = 0;

  /**
   * @brief Check the conditions for the
   * algorithm to run, this is useful when
   * there are strategies, options that
   * contradict the algorithm
   * 
   * @return true 
   * @return false 
   */
  virtual bool can_run(const cmdlinet &cmd)
  {
    for(auto &x : unsupported_options)
    {
      if(cmd.isset(x.c_str()))
      {
        std::cout << x << " cannot be used with the algorithm\n";
        return false;
      }
      std::cout << "Looking for " << x;
    }
    return true;
  }

protected:
  /**
   * Which options if set break the analysis? 
   */
  std::vector<std::string> unsupported_options = {};
};

/**
 * @brief Base interface for goto-programs algorithms
 */
class goto_program_algorithm : public algorithm
{
public:
  explicit goto_program_algorithm(goto_programt &goto_program)
    : goto_program(goto_program)
  {
  }

protected:
  goto_programt &goto_program;
};

/**
 * @brief Base interface for goto-functions algorithms
 */
class goto_functions_algorithm : public algorithm
{
public:
  explicit goto_functions_algorithm(goto_functionst &goto_functions)
    : goto_functions(goto_functions)
  {
  }

protected:
  goto_functionst &goto_functions;
};

#endif //ESBMC_ALGORITHM_H