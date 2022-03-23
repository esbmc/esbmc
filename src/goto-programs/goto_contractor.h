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
#include "goto_functions.h"
#include <util/algorithms.h>
#include <util/message/message.h>
#include <goto-programs/goto_loops.h>
#include <goto-programs/remove_skip.h>
#include <goto-programs/goto_functions.h>
#include <goto-programs/goto_functions.h>
#include <goto-programs/loopst.h>
#include <util/message/message_stream.h>
#include <util/std_types.h>
#include "irep2/irep2.h"
#include "util/type_byte_size.h"

#define MAX_VAR 10

using namespace ibex;

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

      ///keep for later. set initial intervals
      //auto w = symbol.type->get_width();
      /*if(is_signedbv_type(symbol.type))
        add_interval(-pow(2,w-1),pow(2,w-1)-1,n);
      else
        add_interval(0,pow(2,w)-1,n);*/
      n++;
      return n - 1;
    }
    return -1;
  }
  void add_interval(double lb, double ub, int index)
  {
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
    ///Used only for testing. will be removed.
    //    std::cout << "size = " << n << std::endl;
    //    std::cout << "[";
    //    for(int i = 0; i < n; i++)
    //      std::cout << var_name[i] << ":" << intervals[i] << " : ";
    //    std::cout << "]" << std::endl;
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

class goto_contractort : public goto_functions_algorithm
{
  // +constructor
public:
  goto_contractort(
    goto_functionst &_goto_functions,
    const messaget &_message_handler)
    : goto_functions_algorithm(_goto_functions, true)
  {
    message_handler = _message_handler;

    run1();
    if(function_loops.size())
    {
      //initialize map
      map = new MyMap();
      vars = new Variable(MAX_VAR);
      //find properties
      //convert from ESBMC to ibex format
      message_handler.status(
        "1/4 - Parsing asserts to create CSP Constraints.");
      get_constraints(_goto_functions);
      //find intervals -- frama-c
      message_handler.status(
        "2/4 - Parsing assumes to set values for variables intervals.");
      get_intervals(_goto_functions);
      //contract
      message_handler.status("3/4 - Applying contractor.");
      auto new_intervals = contractor();
      //reflect results on goto-program by inserting assume.
      message_handler.status("4/4 - Inserting assumes.");
      insert_assume(_goto_functions, new_intervals);
      //clean up
    }
  }

private:
  IntervalVector domains;
  Variable *vars;
  Ctc *ctc;
  MyMap *map;
  NumConstraint *constraint;

  unsigned number_of_functions = 0;

  typedef std::list<loopst> function_loopst;
  function_loopst function_loops;

  messaget message_handler;

  void get_constraints(goto_functionst functionst);
  void get_intervals(goto_functionst functionst);

  IntervalVector contractor();

  ibex::CmpOp get_complement(ibex::CmpOp);
  void insert_assume(goto_functionst goto_functions, IntervalVector vector);
  std::string get_constraints_from_expr2t(irep_container<expr2t>);

  Ctc *create_contractors_from_expr2t(irep_container<expr2t>);
  NumConstraint *create_constraint_from_expr2t(irep_container<expr2t>);
  Function *create_function_from_expr2t(irep_container<expr2t>);
  int create_variable_from_expr2t(irep_container<expr2t>);

  void parse_intervals(irep_container<expr2t> expr);
  //override to get loops
  bool run1();
};

#endif //ESBMC_GOTO_CONTRACTOR_H
