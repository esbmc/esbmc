/**
 * @file bmc_domain_split.h
 * @version 1.0
 *
 * @section
 *
 * Includes functionally to help domain splitting of a variable.
 * 
 */

#ifndef _BMC_DOMAIN_SPLIT
#define _BMC_DOMAIN_SPLIT

#include <util/message/message_stream.h>
#include <esbmc/bmc.h>
#include <irep2/irep2_expr.h>

class BMC_Domain_Split
{
public:
  /* Constructor for the BMC_Domain_Split class
   *
   * @param msg messaget object for printing
   * @param steps Single-static-assignment steps to find variables in
   */
  BMC_Domain_Split(
    const messaget &msg,
    symex_target_equationt::SSA_stepst &steps)
    : msg(msg), steps(steps)
  {
  }

  //METHODS

  /* Finds the variables in the SSA steps which take a non-deterministic value.
   * These variables are appended to the free_vars vector.
   */
  void bmc_get_free_variables();

  /* Counts the number of times the variables in free_vars vector occurs in the SSA code.
   *
   * This should be run after bmc_get_free_variables.
   */
  void bmc_free_var_occurances();

  /* 
   * Selects the most used variable in var_used.
   *
   * This should only be run after bmc_free_var_occurances
   *
   * @return a pair containing the most used variable and the number of times it has been played.
   */
  std::pair<expr2tc, int> bmc_most_used() const;

  /* Creates expressions that can be used to partition the domain of a given variable. 
   *
   * @param var variable whose domain should be partition.
   * @return A vector of expressions that can be used to split the domain. These should each be inserted into an SMT equation.
   */
  std::vector<expr2tc> bmc_split_domain_exprs(const expr2tc &var) const;

  /* Creates expressions that can be used to partition the domain of a given variable. 
   *
   * @param var variable whose domain should be partition.
   * @param depth the number of times the domain should be partitioned.
   * @return A vector of expressions that can be used to split the domain. These should each be inserted into an SMT equation.
   */
  std::vector<expr2tc>
  bmc_split_domain_exprs(const expr2tc &var, unsigned int depth) const;

  /* Prints the free variables in free_vars
   */
  void print_free_vars() const;

  //FIELDS
private:
  const messaget &msg;
  symex_target_equationt::SSA_stepst &steps;

public:
  std::vector<expr2tc> free_vars;
  std::vector<int> free_var_counts;
};

/* Counts the number of occurances of the symbol v in the expr rhs.
 *
 * @param rhs expression representing the right-hand-side of an assignment.
 * @param v variable expression whose occurances will be counded.
 * @return number of occurances.
 */
int occurances(const expr2tc &rhs, expr2tc &v);

/* Returns points that divide an interval.
 *
 * @param l lower value of the interval.
 * @param h highest value of the inteval.
 * @param d number of pieces interval should be divided into.
 * @return a vector of ints that are the divisions of the interval [l, h]>
 */
std::vector<int> divide_interval(int l, int h, int d);
#endif
