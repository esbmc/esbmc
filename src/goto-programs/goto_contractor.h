#ifndef ESBMC_GOTO_CONTRACTOR_H
#define ESBMC_GOTO_CONTRACTOR_H

#include <iostream>
#include <util/goto_expr_factory.h>
#include <goto-programs/goto_functions.h>
#include <util/algorithms.h>
#include <util/message.h>
#include <goto-programs/goto_loops.h>
#include <goto-programs/remove_skip.h>
#include <goto-programs/goto_functions.h>
#include <goto-programs/goto_functions.h>
#include <goto-programs/loopst.h>
#include <util/std_types.h>
#include <ibex.h>
#include <ibex/ibex_Interval.h>
#include <ibex/ibex_Expr.h>
#include <ibex/ibex_Ctc.h>
#include <irep2/irep2.h>
#include <util/type_byte_size.h>

void goto_contractor(goto_functionst &goto_functions);

class vart
{
private:
  ibex::Interval interval;
  std::string var_name;
  symbol2tc symbol;
  size_t index;
  bool interval_changed;

public:
  int getIndex() const;

public:
  vart(const string &varName, const symbol2tc &symbol, const size_t &index);
  const ibex::Interval &getInterval() const;
  void setInterval(const ibex::Interval &interval);
  bool isIntervalChanged() const;
  void setIntervalChanged(bool intervalChanged);
  const symbol2tc &getSymbol() const;
};
/**
 * This class is for mapping the variables with their names and intervals.
 * It includes functionalities such as search for a variable by name and add
 * a new variable. Also update a variable interval by changing the upper/lower
 * limit.
 */
class CspMap
{
public:
  static constexpr int MAX_VAR = 10;
  static constexpr int NOT_FOUND = -1;

  std::map<std::string, vart> var_map;

  CspMap()
  {
  }
  size_t add_var(const std::string &name, const symbol2t &symbol)
  {
    auto find = var_map.find(name);
    if(find == var_map.end())
    {
      vart var(name, symbol, n);
      var_map.insert(std::make_pair(name, var));

      //TODO: set initial intervals based on type and width.

      n++;
      return n - 1;
    }
    return find->second.getIndex();
  }

  void update_lb_interval(double lb, const std::string &name)
  {
    auto find = var_map.find(name);
    ibex::Interval X(lb, find->second.getInterval().ub());
    find->second.setInterval(X);
  }
  void update_ub_interval(double ub, const std::string &name)
  {
    auto find = var_map.find(name);
    ibex::Interval X(find->second.getInterval().lb(), ub);
    find->second.setInterval(X);
  }
  int find(const std::string &name)
  {
    auto find = var_map.find(name);
    if(find == var_map.end())
      return NOT_FOUND;
    return find->second.getIndex();
  }
  size_t size() const
  {
    return n;
  }

  ibex::IntervalVector create_interval_vector()
  {
    ibex::IntervalVector X(var_map.size());
    for(auto const &var : var_map)
      X[var.second.getIndex()] = var.second.getInterval();
    return X;
  }

  void update_intervals(ibex::IntervalVector vector)
  {
    //check if interval box is empty set or if the interval is degenerated
    // in the case of a single interval
    if(vector.is_empty() || (vector.size() == 1 && vector[0].is_degenerated()))
      is_empty_vector = true;

    for(auto &var : var_map)
    {
      if(var.second.getInterval() != vector[var.second.getIndex()])
      {
        var.second.setInterval(vector[var.second.getIndex()]);
        var.second.setIntervalChanged(true);
      }
    }
  }

private:
  size_t n = 0;
  bool is_empty_vector = false;

public:
  bool is_empty_set() const
  {
    return is_empty_vector;
  }
};

class goto_contractort : public goto_functions_algorithm
{
public:
  /**
   * This constructor will run the goto-contractor procedure.
   * it will go through 4 steps.
   * First is parsing the properties.
   * Second, parsing the intervals.
   * Third, applying the contractor.
   * Fourth, inserting assumes in the program to reflect the contracted intervals.
   * @param _goto_functions
   */
  goto_contractort(goto_functionst &_goto_functions)
    : goto_functions_algorithm(true), goto_functions(_goto_functions)
  {
    initialize_main_function_loops();
    if(!function_loops.empty())
    {
      vars = new ibex::Variable(CspMap::MAX_VAR);
      log_status("1/4 - Parsing asserts to create CSP Constraints.");
      get_constraints(_goto_functions);
      if(constraint == nullptr)
      {
        log_status(
          "Constraint expression not supported. Aborting goto-contractor");
        return;
      }

      log_status(
        "2/4 - Parsing assumes to set values for variables intervals.");
      get_intervals(_goto_functions);

      log_status("3/4 - Applying contractor.");
      contractor();

      log_status("4/4 - Inserting assumes.");
      insert_assume(_goto_functions);
    }
  }

protected:
  goto_functionst &goto_functions;

private:
  ibex::IntervalVector domains;
  ///vars variable references to be used in Ibex formulas
  ibex::Variable *vars;
  /// map is where the variable references and intervals are stored.
  CspMap map;
  /// constraint is where the constraint for CSP will be stored.
  ibex::NumConstraint *constraint = nullptr;

  unsigned number_of_functions = 0;

  typedef std::list<loopst> function_loopst;
  function_loopst function_loops;

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

  /// \Function contractor function will apply the contractor on the parsed
  /// constraint and intervals. it will apply the inner contractor by
  /// calculating the complement of the assert and contract.
  /// \return Interval vector that represents the area that should be checked
  /// by the bmc.
  void contractor();

  /** \Function get_complement will take a comparison operation and get its
   * complement. Operators are defined in ibex as the enumeration ibex::CmpOP.
   * @param CmpOp is a comparison operator.
   * @return complement to to received CmpOP.
   */
  ibex::CmpOp get_complement(ibex::CmpOp);
  /**
   * @function insert_assume is the function that will use the intervals
   * produced by the contractor and compare it to the original intervals.
   * If there are any changes, it will be inserted into the program as assumes
   * in the format of assume(<variable> <operator> <value>) where <variable> is
   * the variable name, <operator> is <=,>= depending if its an upper or a lower
   * limit and <value> is the value of the interval limit. If the resulting
   * interval is empty (check via is_empty_vector flag), it will insert
   * assume(0). It will also search for the last loop in the program based
   * on location.
   * @param goto_functions goto program functions
   * @param vector result from the contractor.
   */
  void insert_assume(goto_functionst goto_functions);

  bool is_unsupported_operator(expr2tc expr);
  ibex::NumConstraint *create_constraint_from_expr2t(irep_container<expr2t>);
  ibex::Function *create_function_from_expr2t(irep_container<expr2t>);
  int create_variable_from_expr2t(irep_container<expr2t>);

  void parse_intervals(irep_container<expr2t> expr);

  bool initialize_main_function_loops();
};
#endif //ESBMC_GOTO_CONTRACTOR_H