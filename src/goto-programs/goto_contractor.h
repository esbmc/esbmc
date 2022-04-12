//
// Created by Mohannad Aldughaim on 09/01/2022.
//

#ifndef ESBMC_GOTO_CONTRACTOR_H
#define ESBMC_GOTO_CONTRACTOR_H

#include <iostream>
#include <util/goto_expr_factory.h>
#include <goto-programs/goto_functions.h>
#include <util/algorithms.h>
#include <util/message/message.h>
#include <goto-programs/goto_loops.h>
#include <goto-programs/remove_skip.h>
#include <goto-programs/goto_functions.h>
#include <goto-programs/goto_functions.h>
#include <goto-programs/loopst.h>
#include <util/message/message_stream.h>
#include <util/std_types.h>
#include <ibex.h>
#include <ibex/ibex_Interval.h>
#include <ibex/ibex_Expr.h>
#include <ibex/ibex_Ctc.h>
#include <irep2/irep2.h>
#include <util/type_byte_size.h>


/**
 * This class is for mapping the variables with their names and intervals.
 * It includes functionalities such as search for a variable by name and add
 * a new variable. Also update a variable interval by changing the upper/lower
 * limit.
 */
class CspMap
{
public:
  static constexpr int MAX_VAR=10;
  static constexpr int NOT_FOUND=-1;

  ibex::IntervalVector intervals;
  std::string var_name[MAX_VAR];
  symbol2tc symbols[MAX_VAR];
  bool nondecreasing[MAX_VAR], nonincreasing[MAX_VAR];

  CspMap()
  {
    intervals.resize(MAX_VAR);
    for(int i=0;i<MAX_VAR;i++)
    nondecreasing[i] = nonincreasing[i] = false;
  }
  int add_var(std::string name, symbol2t symbol)
  {
    if(find(name) == NOT_FOUND && n < MAX_VAR)
    {
      symbols[n] = symbol;
      var_name[n] = name;

      //TODO: set initial intervals based on type and width.

      n++;
      return n - 1;
    }
    return NOT_FOUND;
  }
  void add_interval(double lb, double ub, int index)
  {
    intervals[index] = ibex::Interval(lb, ub);
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
    for(size_t i = 0; i < n; i++)
      if(var_name[i] == name)
        return i;

    return NOT_FOUND;
  }
  size_t size()
  {
    return n;
  }

private:
  size_t n = 0;
  ibex::Variable *x;
};

class goto_contractort : public goto_functions_algorithm
{
  // +constructor

public:
  /**
   * This constructor will run the goto-contractor procedure.
   * it will go through 4 steps.
   * First is parsing the properties.
   * Second, parsing the intervals.
   * Third, applying the contractor.
   * Fourth, inserting assumes in the program to reflect the contracted intervals.
   * @param _goto_functions
   * @param _message_handler
   */
  goto_contractort(
    goto_functionst &_goto_functions,
    const messaget &_message_handler)
    : goto_functions_algorithm(_goto_functions, true)
  {
    message_handler = _message_handler;
    initialize_main_function_loops();
    if(function_loops.size())
    {
      map = new CspMap();
      vars = new ibex::Variable(CspMap::MAX_VAR);
      message_handler.status(
        "1/5 - Parsing asserts to create CSP Constraints.");
      get_constraints(_goto_functions);
      if(constraint == nullptr)
      {
        message_handler.status(
          "Constraint expression not supported. Aborting goto-contractor");
        return;
      }
      message_handler.status("2/5 - Checking if loop is monotone.");
      monotonicity_check();
      message_handler.status(
        "3/5 - Parsing assumes to set values for variables intervals.");
      get_intervals(_goto_functions);
      message_handler.status("4/5 - Applying contractor.");
      auto new_intervals = contractor();
      if(new_intervals == map->intervals)
      {
        message_handler.status(
          "5/5 - Intervals remained unchanged. No assumes will be inserted");
      }
      else
      {
        message_handler.status("5/5 - Inserting assumes.");
        insert_assume(_goto_functions, new_intervals);
      }
    }
  }

private:
  ibex::IntervalVector domains;
  ///vars variable references to be used in Ibex formulas
  ibex::Variable *vars;
  ibex::Ctc *ctc;
  /// map is where the variable references and intervals are stored.
  CspMap *map;
  /// constraint is where the constraint for CSP will be stored.
  std::shared_ptr<ibex::NumConstraint> constraint;

  unsigned number_of_functions = 0;

  typedef std::list<loopst> function_loopst;
  function_loopst function_loops;

  messaget message_handler;

  /// \Function get_constraint is a function that will go through each asert
  /// in the program and parse it from ESBMC expression to an IBEX expression
  /// that will be added to constraints in the CSP. the function will return
  /// nothing. However the constraints be added to the list of constraints.
  /// \param functionst list of functions in the goto program
  void get_constraints(goto_functionst functionst);

  /// \Function get_intervals is a function that will go through each asert in
  /// the program and parse it from ESBMC expression to a triplet that are the
  /// variable name and and update its interval depending on the relation it
  /// will decide if the lower or the upper limit or both. the function will
  /// return nothing. However the values of the intervals of each variable will
  /// be updated in the Map that holds the variable information.
  /// \param functionst list of functions in the goto program
  void get_intervals(goto_functionst functionst);

  /// \Function contractor function will apply the contractor on the parsed constraint and intervals. it will apply the inner contractor by calculating the complement of the assert and contract.
  /// \return Interval vector that represents the area that should be checked by the bmc.
  ibex::IntervalVector contractor();

  /** \Function get_complement will take a comparison operation and get its complement. Operators are defined in ibex as the enumeration ibex::CmpOP.
   * @param CmpOp is a comparison operator.
   * @return complement to to received CmpOP.
   */
  ibex::CmpOp get_complement(ibex::CmpOp);
  /**
   * @function insert_assume is the function that will use the intervals
   * produced by the contractor and compare it to the original intervals.
   * If there are any changes, it will be inserted into the program as assumes
   * in the format of assume(<variable> <operator> <value>) where variable is
   * the variable name, operator is <=,>= depending if its an upper or a lower
   * limit and value is the value of the interval limit.
   * @param goto_functions goto program functions
   * @param vector result from the contractor.
   */
  void
  insert_assume(goto_functionst goto_functions, ibex::IntervalVector vector);

  ibex::Ctc *create_contractors_from_expr2t(irep_container<expr2t>);
  std::shared_ptr<ibex::NumConstraint>
    create_constraint_from_expr2t(irep_container<expr2t>);
  std::shared_ptr<ibex::Function>
    create_function_from_expr2t(irep_container<expr2t>);
  int create_variable_from_expr2t(irep_container<expr2t>);

  void parse_intervals(irep_container<expr2t> expr);

  bool initialize_main_function_loops();

  /**
 * Naive appraoch to determine if a loop is monotonic
 * This function will be replaced later with exrapolation by using intervals
 * from frama-c eva plugin
 * @param functionst
 */
  void monotonicity_check(void);
};

#endif //ESBMC_GOTO_CONTRACTOR_H
