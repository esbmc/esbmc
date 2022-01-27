//
// Created by Mohannad Aldughaim on 09/01/2022.
//

#ifndef ESBMC_GOTO_CONTRACTOR_H
#define ESBMC_GOTO_CONTRACTOR_H

#include "goto_k_induction.h"
#include <iostream>
#include <ibex/ibex_Interval.h>
#include "ibex.h"

using namespace ibex;

void goto_contractor(
  goto_functionst &goto_functions,
  const messaget &message_handler);

class goto_contractort : public goto_k_inductiont
{
  // +constructor
public:
  goto_contractort(
    const irep_idt &_function_name,
    goto_functionst &_goto_functions,
    goto_functiont &_goto_function,
    const messaget &_message_handler)
    : goto_k_inductiont(
        _function_name,
        _goto_functions,
        _goto_function,
        _message_handler)
  {
    if(function_loops.size())
    {
      //TODO: find properties -- goto_function
      //TODO: find intervals -- frama-c
      get_intervals(_goto_functions);
      //TODO: convert from ESBMC to ibex format

      //TODO: find goto-program - done
      //TODO: add IBex library - done
      //TODO: contract - done
      //auto new_intervals = contractor(n_vars, vars, domains, constraint);
      //TODO: reflect results on goto-program by inserting assume.
      insert_assume(_goto_functions);
    }
  }

private:
  int n_vars;
  std::string **vars;
  IntervalVector domains;
  std::string *constraint;

  //void goto_k_induction();
  void get_intervals(goto_functionst functionst);
  IntervalVector contractor(
    int n_vars,
    string **vars,
    IntervalVector domains,
    string *constraint);
  void insert_assume(goto_functionst goto_functions);
};

#endif //ESBMC_GOTO_CONTRACTOR_H
