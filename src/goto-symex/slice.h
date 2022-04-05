/*******************************************************************\

Module: Slicer for symex traces

Author: Daniel Kroening, kroening@kroening.com

Contributors:
 - Rafael Sá Menezes, 2022

\*******************************************************************/

#ifndef CPROVER_GOTO_SYMEX_SLICE_H
#define CPROVER_GOTO_SYMEX_SLICE_H

#include <goto-symex/renaming.h>
#include <goto-symex/symex_target_equation.h>
#include <unordered_set>

namespace slicer
{
/**
 * Helper function to call the slicer
 * @param eq symex formula to be sliced
 * @param slice_assume whether assumes should be sliced
 * @param ignored_symbols list of symbols that cannot be sliced
 * @return number of steps that were ignored
 */
BigInt slice(
  std::shared_ptr<symex_target_equationt> &eq,
  bool slice_assume,
  std::unordered_set<std::string> ignored_symbols);

/**
 * Naive slicer: slice every step after the last assertion
 * @param eq symex formula to be sliced
 * @return number of steps that were ignored
 */
BigInt simple_slice(std::shared_ptr<symex_target_equationt> &eq);
} // namespace slicer
/**
 * @brief Class for the symex-slicer, this slicer is to be executed
 * on SSA formula in order to remove every symbol that does not depends
 * on it
 *
 * It works by constructing a symbol dependency list by transversing
 * the SSA formula in reverse order. If any assume, assignment, or renumber
 * step does not belong into this dependency, then it will be ignored.
 */
class symex_slicet
{
public:
  symex_slicet(bool assume, std::unordered_set<std::string> ignored_symbols);
  /**
   * Iterate over all steps of the \eq in REVERSE order,
   * getting symbol dependencies. If an
   * assignment, renumber or assume does not contain one
   * of the dependency symbols, then it will be ignored.
   *
   * @param eq symex formula to be sliced
   */
  void slice(std::shared_ptr<symex_target_equationt> &eq);
  // To show how many assignments were sliced
  BigInt ignored;

protected:
  // Option to enable slicing of assumes
  bool slice_assumes;
  /**
   * This type will be the one used to hold every symbol
   * that the current equation depends on.
   */
  typedef std::unordered_set<std::string> symbol_sett;
  symbol_sett depends;
  const symbol_sett ignored_symbols;
  // Anonymous function to add elements into #depends
  std::function<bool(const symbol2t &)> add_to_deps;
  // TODO: we probably don't need #add_to_deps
  // TODO: In the implementation, there is also the #check_in_deps which we could remove

  /**
   * Recursively explores the operands of an expression \expr
   * If a symbol is found, then it is added into the #depends
   * member.
   *
   * TODO: We probably don't need to pass the \param
   *
   * @param expr expression to extract every symbol
   * @param fn `add_to_depends` kind of function.
   * @return true if at least one symbol was found
   */
  bool
  get_symbols(const expr2tc &expr, std::function<bool(const symbol2t &)> fn);

  /**
   * Helper function, it is used to select specialization will
   * be used, i.e. assume, assignment or renumber
   *
   * Note 1: ASSERTS are not sliced, only their symbols are added
   * into the #depends
   *
   * Note 2: Similar to ASSERTS, if 'slice-assumes' option is
   * is not enabled. Then only its symbols are added into the
   * #depends
   *
   * TODO: All slice specialization can be converted into a lambda
   *
   * @param SSA_step any kind of SSA expression
   */
  void slice(symex_target_equationt::SSA_stept &SSA_step);

  /**
   * Remove unneeded assumes from the formula
   *
   * Check if the Assume cond symbol is in the #depends, if
   * it is not then mark the \SSA_Step as ignored.
   *
   * If the assume cond is in the #depends, then add its guards
   * and cond into the #depends
   *
   * Note 1: All the conditions operands are going to be added
   * into the #depends. This makes that the condition itself as
   * a "reverse taint"
   *
   * TODO: What happens if the ASSUME would result in false?
   *
   * @param SSA_step an assume step
   */
  void slice_assume(symex_target_equationt::SSA_stept &SSA_step);

  /**
   * Remove unneeded assignments from the formula
   *
   * Check if the LHS symbol is in the #depends, if
   * it is not then mark the \SSA_Step as ignored.
   *
   * If the assume cond is in the #depends, then add its guards
   * and cond into the #depends
   *
   * @param SSA_step an assignment step
   */
  void slice_assignment(symex_target_equationt::SSA_stept &SSA_step);

  /**
   * Remove unneeded renumbers from the formula
   *
   * Check if the LHS symbol is in the #depends, if
   * it is not then mark the \SSA_Step as ignored.
   *
   * If the assume cond is in the #depends, then add its guards
   * and cond into the #depends
   *
   * @param SSA_step an renumber step
   */
  void slice_renumber(symex_target_equationt::SSA_stept &SSA_step);
};

#endif
