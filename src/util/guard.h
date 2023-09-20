#ifndef CPROVER_GUARD_H
#define CPROVER_GUARD_H

#include <util/expr.h>
#include <irep2/irep2.h>
#include <util/migrate.h>

/**
 * Defines a guard. A guard is the antecedent (i.e. left-hand side) of an
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
 *
 * It is important to note that even though conceptually the set of conditions
 * could indeed be an (unordered) set, the order of the conditions put into
 * guards actually matters. From <https://github.com/esbmc/esbmc/pull/1297>:
 *
 *   The SMT backend relies on the order of operands to cache the converted
 *   expressions, you'll see huge slowdowns if the order of the guards is not
 *   stable (we should document it somewhere...).
 *
 * The \ref guardt class internally maintains two separate representations of
 * the sequence of conditions: the `guard_list` and the `g_expr`. These two are
 * kept in sync. Formulated as an invariant, the following always hold:
 * 1. `!g_expr` iff `guard_list.empty()`
 * 2. if `!guard_list.empty()` then `g_expr` is equivalent to the conjunction of
 *    all conditions in `guard_list`
 */
class guardt
{
public:
  typedef std::vector<expr2tc> guard_listt;

  guardt() noexcept = default;

  /** \brief Adds a boolean condition to the end of this guard. */
  void add(const expr2tc &expr);

  /** \brief Appends all conditions from a guard to the end of this guard. */
  void append(const guardt &guard);

  /**
   * Returns a formula equivalent to the conjunction of all conditions.
   *
   * Does not have to be an and2tc expression, e.g., in case the guard just
   * contains a single formula, or a known-false formula, or in case it is
   * empty. */
  expr2tc as_expr() const;

  /**
   * Transforms the given expression 'dest' to a formula that is equivalent to
   * the implication (G => dest) where G is this guard. */
  void guard_expr(expr2tc &dest) const;

  bool disjunction_may_simplify(const guardt &other_guard) const;
  bool is_true() const;
  bool is_false() const;

  void make_true();
  void make_false();
  void swap(guardt &g);

  /**
   * \brief Difference between two guards.
   *
   * Let P be the common prefix of the two guard-lists in terms of the
   * expressions being exact copies (in contrast to equivalent formulas like
   * not2tc(not2tc(e)) and e).
   *
   * Then the difference between g1 and g2 is defined to be the conjunction of
   * the conditions in g1 following P, that is, the prefix P is removed from g1.
   */
  friend guardt &operator-=(guardt &g1, const guardt &g2);

  /** \brief Disjunction of two guards.
   *
   * Keeps the prefix common to g1 and g2 unchanged and adds a disjunction of
   * the remaining conditions in g1 and g2 to g1.
   *
   * \param [in,out] g1 guard set to the disjunction of g1 and g2
   * \param [in]     g2 guard to build the disjunction of g1 with
   * \return a reference to the modified g1
   */
  friend guardt &operator|=(guardt &g1, const guardt &g2);
  friend bool operator==(const guardt &g1, const guardt &g2);

  void dump() const;

protected:
  guard_listt guard_list;
  expr2tc g_expr;

  explicit guardt(guard_listt guard_list) noexcept;

  bool is_single_symbol() const;
  void clear();

  void build_guard_expr();
};

#endif
