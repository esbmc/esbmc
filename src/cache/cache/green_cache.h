//
// Created by rafaelsa on 10/03/2020.
//

#ifndef ESBMC_GREEN_CACHE_H
#define ESBMC_GREEN_CACHE_H

#include <cache/ssa_step_algorithm.h>
#include <cache/containers/crc_set_container.h>

/**
 * A Green implementation for SSA steps
 *
 * Based on the paper: "Green: Reducing, Reusing and Recycling Constraint in
 * Program analysis"
 *
 * Green algorithm consists in:
 *
 * 1. Slice the original formula
 * 2. Canonization: Puts the formula in a normal form
 * 3. Recover: Checks formula on database and slices it
 * 4. Translate: Transform the formula into SAT/SMT form
 * 5. Storage: Save formula results
 *
 * 1 and 4 I will assume that esbmc already does it.
 * TODO: add a renaming algorithm
 * TODO: create a constant propagation expr algorithm
 *
 */
class green_cache : public ssa_step_algorithm_hidden
{
public:
  green_cache(
    symex_target_equationt::SSA_stepst &steps,
    bool apply_reordering = true,
    bool apply_renaming = true,
    bool apply_normalization = true)
    : ssa_step_algorithm_hidden(steps),
      apply_reordering(apply_reordering),
      apply_renaming(apply_renaming),
      apply_normalization(apply_normalization)
  {
  }

  /**
   * After this is used, all guards and expressions of the current SSA are marked
   * as unsat adding it to a ssa_container.
   */
  void mark_ssa_as_unsat();

protected:
  /**
   * A map of guards referencing their expressions.
   *
   * This is used to flatten an assertive
   *
   * Example:
   *
   * guard1 -> x == 1
   * guard2 -> x < 10
   *
   * If an assertive in the format
   *
   * guard3 -> guard1 && guard2 -> !(x == 0)
   * using this dictionary we can check what is guard1 and guard2
   */
  std::unordered_map<std::string, crc_expr> items;

  /**
   * Convert a simple guard expression into a set of the RHS expressions
   *
   * Input:
   *  a simple guard in the format of:
   *
   * guard -> expr
   *
   * This should simple extract expr from it
   *
   * @param expr
   * @return
   */
  static crc_expr parse_guard(const expr2tc &expr);

  /**
   * Convert a complex guard expression into a set of the RHS expressions
   *
   * Input:
   *  a guard in the format of:
   *
   * guard -> guard && guard && guard ... && guard -> !expr
   *
   * This should simple extract expr from it
   *
   * @param expr
   * @return
   */
  void parse_implication_guard(const expr2tc &expr, crc_expr &inner_items);

  /**
   * Simple take a expression and apply a hashing algorithm
   * @param expr
   * @return
   */
  static crc_hash convert_expr_to_hash(const expr2tc &expr);

  /**
   * Takes an expression and apply the green heuristics
   * 1. Reordering
   * 2. Put in normal form
   * TODO: 3. Rename it
   * @param expr
   */
  void canonize_expr(expr2tc &expr);

  /**
   * Loads a ssa_container with all expressions known to be unsat
   */
  void load_unsat_container();

  void run_on_assignment(symex_target_equationt::SSA_stept &step) override;
  void run_on_assert(symex_target_equationt::SSA_stept &step) override;

  // Not used
  void run_on_assume(symex_target_equationt::SSA_stept &step) override
  {
  }
  void run_on_output(symex_target_equationt::SSA_stept &step) override
  {
  }
  void run_on_skip(symex_target_equationt::SSA_stept &step) override
  {
  }
  void run_on_renumber(symex_target_equationt::SSA_stept &step) override
  {
  }

private:
  const bool apply_reordering;
  const bool apply_renaming;
  const bool apply_normalization;

  ssa_set_container unsat_container;
  // TODO: Add sat_container
};

#endif //ESBMC_GREEN_CACHE_H
