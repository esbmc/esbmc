#ifndef CPROVER_GUARD_H
#define CPROVER_GUARD_H

#include <util/expr.h>
#include <irep2/irep2.h>
#include <util/migrate.h>

/* Defines a guard. A guard is the antecedent (i.e. left-hand side) of an
 * implication. This guard implementation is even more concrete in that on the
 * top level it is interpreted as a conjunction of conditions.
 *
 * Thus, add() and append() are meant to enlarge the set of conditions.
 *
 * A condition could be an arbitrary boolean expression, including disjunctions,
 * etc.
 *
 * As guards are quite ubiquitous, there sometimes is a need to know whether the
 * represented conjunction is statically evaluable, in particular whether it is
 * known to be constant true or constant false. At present, these
 * properties are obtained purely syntactically (e.g. an empty set of conditions
 * is known to be true).
 *
 * In addition, these guards provide operator support for building
 * - the disjunction of two guards via |=, and
 * - the "difference" of two guards via -=; see there for more details.
 */
class guardt
{
public:
  typedef std::vector<expr2tc> guard_listt;

  void add(const expr2tc &expr);
  void append(const guardt &guard);

  /* Returns a formula equivalent to the conjunction of all conditions.
   *
   * Does not have to be an and2tc expression, e.g., in case the guard just
   * contains a single formula, or a known-false formula, or in case it is
   * empty. */
  expr2tc as_expr() const;

  /* Transforms the given expression 'dest' to a formula that is equivalent to
   * the implication (G => dest) where G is this guard. */
  void guard_expr(expr2tc &dest) const;

  bool disjunction_may_simplify(const guardt &other_guard) const;
  bool is_true() const;
  bool is_false() const;

  void make_true();
  void make_false();
  void swap(guardt &g);

  /* Difference between two guards.
   *
   * It is defined as those conditions in g1 that are not present in g2, where
   * present means that they are exact copies (in contrast to equivalent
   * formulas like not2tc(not2tc(e)) and e). */
  friend guardt &operator-=(guardt &g1, guardt g2);

  /* Disjunction of two guards. */
  friend guardt &operator|=(guardt &g1, const guardt &g2);
  friend bool operator==(const guardt &g1, const guardt &g2);

  void dump() const;

protected:
  guard_listt guard_list;
  expr2tc g_expr;

  bool is_single_symbol() const;
  void clear();

  void build_guard_expr();
};

#endif
