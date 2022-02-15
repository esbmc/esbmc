//
// Created by Mohannad Aldughaim on 09/01/2022.
//

#ifndef ESBMC_GOTO_CONTRACTOR_H
#define ESBMC_GOTO_CONTRACTOR_H

#include "goto_k_induction.h"
#include <iostream>
#include <ibex/ibex_Interval.h>
#include "ibex.h"
#include "util/goto_expr_factory.h"

#define MAX_VAR 10

using namespace ibex;

void goto_contractor(
  goto_functionst &goto_functions,
  const messaget &message_handler);

class MyMap
{
public:
  IntervalVector intervals;
  std::string var_name[MAX_VAR];
  symbol2tc symbols[MAX_VAR];

  MyMap()
  {
    intervals.resize(MAX_VAR);
  }
  int add_var(std::string name, symbol2t symbol)
  {
    if(find(name) == -1 && n < MAX_VAR)
    {
      symbols[n] = symbol;
      var_name[n] = name;
      auto w = symbol.type->get_width();
      if(is_signedbv_type(symbol.type))
        add_interval(pow(-2,w-1),pow(2,w-1)-1,n);
      else
        add_interval(0,pow(2,w)-1,n);
      n++;
      return n - 1;
    }
    return -1;
  }
  void add_interval(double lb, double ub, int index)
  {
    interval *p;
    intervals[index] = Interval(lb, ub);
  }
  void update_lb_interval(double lb, int index)
  {
    add_interval(lb, intervals[index].ub(), index);
  }
  void update_ub_interval(double ub, int index)
  {
    add_interval(intervals[index].lb(), ub, index);
  }
  int find(std::string name)
  {
    for(int i = 0; i < n; i++)
      if(var_name[i] == name)
        return i;

    return -1;
  }
  void dump()
  {
    std::cout << "size = " << n << std::endl;
    std::cout << "[";
    for(int i = 0; i < n; i++)
      std::cout << var_name[i] << ":" << intervals[i] << " : ";
    std::cout << "]" << std::endl;
  }
  int size()
  {
    return n;
  }

private:
  int n = 0;
  //Variable *x[10];
  Variable *x;
};

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
      map = new MyMap();
      vars = new Variable(MAX_VAR);
      //TODO: find properties -- goto_function
      get_constraints(_goto_functions);
      //TODO: find intervals -- frama-c
      //TODO: convert from ESBMC to ibex format
      get_intervals(_goto_functions);
      //TODO: find goto-program - done
      //TODO: add IBex library - done
      //TODO: contract - done
      auto new_intervals = contractor();
      //TODO: reflect results on goto-program by inserting assume.
      insert_assume(_goto_functions, new_intervals);
    }
  }

private:
  IntervalVector domains;
  Variable *vars;
  Ctc *ctc;
  MyMap *map;
  NumConstraint *constraint;

  //void goto_k_induction();
  void get_constraints(goto_functionst functionst);
  void get_intervals(goto_functionst functionst);

  IntervalVector contractor();

  void insert_assume(goto_functionst goto_functions, IntervalVector vector);
  std::string get_constraints_from_expr2t(irep_container<expr2t>);

  Ctc *create_contractors_from_expr2t(irep_container<expr2t>);
  NumConstraint *create_constraint_from_expr2t(irep_container<expr2t>);
  Function *create_function_from_expr2t(irep_container<expr2t>);
  int create_variable_from_expr2t(irep_container<expr2t>);

  void parse_intervals(irep_container<expr2t> expr);
};

#endif //ESBMC_GOTO_CONTRACTOR_H
