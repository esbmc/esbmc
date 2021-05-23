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
#include <util/message.h>
/**
 * @brief Base interface to run an algorithm in esbmc
 */
class algorithm
{
public:
  algorithm(std::string disable_cmd, bool sideeffect)
    : disable_command(disable_cmd), sideeffect(sideeffect)
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
   * @brief Check if algorithm can be
   * executed and run it
   */
  bool check_and_run(const cmdlinet &cmd)
  {
    if(can_run(cmd))
      return run();
    return false;
  }

  /**
   * @brief Check the conditions for the
   * algorithm to run, this is useful when
   * there are strategies or options that
   * contradict the algorithm
   * 
   * @return true 
   * @return false 
   */
  virtual bool can_run(const cmdlinet &cmd)
  {
    if(cmd.isset(disable_command.c_str()))
      return false;
    for(auto &x : unsupported_options)
      if(cmd.isset(x.c_str()))
        return false;
    return true;
  }

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
  // Which options if set break the analysis?
  std::vector<std::string> unsupported_options = {};
  // Every algorithm should have a way to be disabled
  std::string &disable_command;
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
    message_handlert &msg,
    std::string disable_cmd,
    bool sideffect)
    : algorithm(disable_cmd, sideffect),
      goto_functions(goto_functions),
      msg(msg)
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

private:
  message_handlert &msg; // This is needed to get the program loop
  unsigned number_of_functions = 0;
  unsigned number_of_loops = 0;
};

#endif //ESBMC_ALGORITHM_H