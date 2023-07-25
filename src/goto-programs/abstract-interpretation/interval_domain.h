/// \file
/// Interval Domain

#ifndef CPROVER_ANALYSES_INTERVAL_DOMAIN_H
#define CPROVER_ANALYSES_INTERVAL_DOMAIN_H

#include <goto-programs/abstract-interpretation/ai.h>
#include <goto-programs/abstract-interpretation/interval_template.h>
#include <goto-programs/abstract-interpretation/wrapped_interval.h>
#include <boost/serialization/nvp.hpp>
#include <util/ieee_float.h>
#include <irep2/irep2_utils.h>
#include <util/mp_arith.h>
#include <boost/multiprecision/cpp_bin_float.hpp>
typedef interval_templatet<BigInt> integer_intervalt;
using real_intervalt =
  interval_templatet<boost::multiprecision::cpp_bin_float_100>;

/**
 * @brief Trivial, conjunctive interval domain for both float
 *        and integers. The categorization 'float' and 'integers'
 *        is done by is_int and is_float.
 */
class interval_domaint : public ai_domain_baset
{
public:
  interval_domaint() : bottom(true)
  {
  }

  void transform(
    goto_programt::const_targett from,
    goto_programt::const_targett to,
    ai_baset &ai,
    const namespacet &ns) final override;

  void output(std::ostream &out) const override;

  static void set_options(const optionst &options);

  void dump() const;

  // TODO: Add options for ai.h
  // Extensions
  static bool
    enable_interval_arithmetic; /// Enable simplification for arithmetic operators
  static bool
    enable_interval_bitwise_arithmetic; /// Enable simplfication for bitwise opeations
  static bool
    enable_modular_intervals; /// Make a modular operation after every assignment
  static bool
    enable_assertion_simplification; /// Simplify condition and assertions with the intervals
  static bool
    enable_contraction_for_abstract_states; /// Use contractor for <= operations
  static bool
    enable_wrapped_intervals; /// Enabled wrapped intervals (disables Integers)
  static bool
    enable_real_intervals; /// Enabled wrapped intervals (disables Integers)

  // Widening options
  static unsigned
    fixpoint_limit; /// Sets a limit for number of iteartions before widening
  static bool
    widening_under_approximate_bound; /// Whether to considers overflows for Integers
  static bool
    widening_extrapolate; /// Extrapolate bound to infinity based on previous iteration
  static bool widening_narrowing; /// Interpolate bound back after fixpoint

  typedef std::unordered_map<irep_idt, integer_intervalt, irep_id_hash>
    int_mapt;

  typedef std::unordered_map<irep_idt, real_intervalt, irep_id_hash> real_mapt;
  typedef std::unordered_map<irep_idt, wrapped_interval, irep_id_hash>
    wrap_mapt;

  typedef std::unordered_map<irep_idt, unsigned, irep_id_hash> fixpoint_counter;

  int_mapt get_int_map() const
  {
    return int_map;
  }

  wrap_mapt get_wrap_map() const
  {
    return wrap_map;
  }

  // Compute whether `cond` is a tautology for the abstract state
  bool forward_check(const expr2tc &cond);

protected:
  /**
  * Sets *this to the mathematical join between the two domains. This can be
  * thought of as an abstract version of union; *this is increased so that it
  * contains all of the values that are represented by b as well as its original
  * intervals. The result is an overapproximation, for example:
  * "[0,1]".join("[3,4]") --> "[0,4]" includes 2 which isn't in [0,1] or [3,4].
  *
  *          Join is used in several places, the most significant being
  *          merge, which uses it to bring together two different paths
  *          of analysis.
  * @param b: The interval domain, b, to join to this domain.
  * @return True if the join increases the set represented by *this, False if
  *   there is no change.
  */
  bool join(const interval_domaint &b);

public:
  bool merge(
    const interval_domaint &b,
    goto_programt::const_targett,
    goto_programt::const_targett)
  {
    return join(b);
  }

  void clear_state()
  {
    int_map.clear();
    real_map.clear();
    wrap_map.clear();
    fixpoint_map.clear();
  }

  // no states
  void make_bottom() final override
  {
    clear_state();
    bottom = true;
  }

  // all states
  void make_top() final override
  {
    clear_state();
    bottom = false;
  }

  void make_entry() final override
  {
    make_top();
  }

  bool is_bottom() const override final
  {
    return bottom;
  }

  bool is_top() const override final
  {
    return !bottom && int_map.empty() && real_map.empty();
  }

  /**
   * @brief Creates an expression with the intervals found for the given symbol
   *
   * For example, if given a symbol expression with the variable 'FOO'.
   * If at the current abstract state the interval for it is of: [2, 42]
   *
   * This would return an expression of the form: AND (>= FOO 2) (<= FOO 42)
   *
   * If top, it will return a true expr.
   * If bottom, it will return a false expr.
   *
   * @param symbol to construct the interval
   * @return expr2tc
   */
  expr2tc make_expression(const expr2tc &symbol) const;

  /**
   * @brief Adds a restriction over all intervals.
   *
   * Do not mistake this by an ASSUME instruction! This can take any type of expression!
   */
  void assume(const expr2tc &);

  /**
   * @brief Uses the abstract state to simplify a given expression using context-
   * specific information.
   * @param parameters: The expression to simplify.
   * @return A simplified version of the expression.
   *
   * This implementation is aimed at reducing assertions to true, particularly
   * range checks for arrays and other bounds checks.
   *
   * Rather than work with the various kinds of exprt directly, we use assume,
   * join and is_bottom.  It is sufficient for the use case and avoids duplicating
   * functionality that is in assume anyway.
   *
   * As some expressions (1<=a && a<=2) can be represented exactly as intervals
   * and some can't (a<1 || a>2), the way these operations are used varies
   * depending on the structure of the expression to try to give the best results.
   * For example negating a disjunction makes it easier for assume to handle.
   */
  virtual bool
  ai_simplify(expr2tc &condition, const namespacet &ns) const override;

protected:
  // Abstract state information
  /// Is this state a bottom. I.e., there is a contradiction between an assignment and an assume
  bool bottom;
  /// Map for all integers intervals
  int_mapt int_map;
  /// Map for all real intervals
  real_mapt real_map;
  /// Map for all wrap intervals
  wrap_mapt wrap_map;
  /// Map for all fixpoint counters
  fixpoint_counter fixpoint_map;

  /**
   * @brief Recursively explores an Expression until it reaches a symbol. If the
   * symbol is a BV, then removes it from the int_map
   *
   * TODO: There are a lot of expressions that are not supported
   * TODO: A utility function that recursively extracts all the symbols of an expr would be very useful
   * @param expr
   */
  void havoc_rec(const expr2tc &expr);

  /**
   * @brief Recursively explores intervals of an assume expression
   *
   * This is the entry-point of the function, it does the following:
   * 1. If the current negation flag is set. Then the operator is inverted (e.g. lower becomes greater_eq)
   * 2. For if2t and or2t... de morgan is applied and this function is called again for its operands
   *
   * @param expr
   * @param negation sets whether the current expr is a negation
   */
  void assume_rec(const expr2tc &expr, bool negation = false);

  /**
   * @brief Recursively explores intervals over a comparation expression, inserting them into int_map
   *
   * @param lhs
   * @param id
   * @param rhs
   */
  void assume_rec(const expr2tc &lhs, expr2t::expr_ids id, const expr2tc &rhs);

  /**
   * @brief Computes an assignment expression.
   *
   * This recomputes the interval of the LHS to be of RHS, i.e. LHS == RHS.
   *
   * @param assignment
   */
  void assign(const expr2tc &assignment);

  /**
   * @brief Applies LHS <= RHS and RHS <= LHS from assignment instructions
   *
   * This is separate from the usual assume_rec as LHS symbol may be inside RHS
   *
   * @tparam Interval interval template specialization (Integers, Reals)
   * @param lhs
   * @param rhs
   */
  template <class Interval>
  void apply_assignment(const expr2tc &lhs, const expr2tc &rhs);

  /**
   * @brief Applies Extrapolation widening algorithm
   *
   * Given two intervals: (a0, b0) (before the computation) and (a1, b1) (after the computation):
   *
   * Widening((a0,b0), (a1,b1)) = (a1 < a0 ? -infinity : a0, b1 > b0 ? infinity : b0 )
   * @tparam Interval interval template specialization (Integers, Reals)
   * @param lhs
   * @param rhs
   */
  template <class Interval>
  Interval extrapolate_intervals(const Interval &before, const Interval &after);

  /**
   * @brief Applies Interpolation narrowing algorithm
   *
   * Given two intervals: (a0, b0) (before the computation) and (a1, b1) (after the computation):
   *
   * Narrowing((a0,b0), (a1,b1)) = (a1 > a0 ? a0 : a1, b1 < b0 ? b0 : b1 )
   * @tparam Interval interval template specialization (Integers, Reals)
   * @param lhs
   * @param rhs
   */
  template <class Interval>
  Interval interpolate_intervals(const Interval &before, const Interval &after);

  /**
   * @brief Applies  LHS < RHS
   *
   * @tparam Interval interval template specialization (Integers, Reals)
   * @param lhs
   * @param rhs
   * @param less_than_equal if operation should be <=
   */
  template <class Interval>
  void apply_assume_less(const expr2tc &lhs, const expr2tc &rhs);

  /**
   * @brief Generates interval with [min, max] using symbol type
   *
   * @tparam Interval interval template specialization (Integers, Reals)
   * @param sym
   * @return Interval the returned interval is either [min, max] or (-infinity, infinity)
   */
  template <class Interval>
  Interval generate_modular_interval(const symbol2t sym) const;

  /**
   * @brief Get the interval for expression
   *
   * This computes the interval of a given expression and returns it
   *
   * @tparam Interval interval template specialization (Integers, Reals)
   * @param e
   * @return Interval
   */
  template <class Interval>
  Interval get_interval(const expr2tc &e) const;

  /**
   * @brief Get the interval from symbol object or top
   *
   * @tparam Interval interval template specialization (Integers, Reals)
   * @param sym
   * @return Interval
   */
  template <class Interval>
  Interval get_interval_from_symbol(const symbol2t &sym) const;

  /**
   * @brief Get the interval from constant expression
   *
   * @tparam Interval interval template specialization (Integers, Reals)
   * @param sym
   * @return Interval
   */
  template <class Interval>
  Interval get_interval_from_const(const expr2tc &sym) const;

  template <class Interval>
  Interval get_top_interval_from_expr(const expr2tc &sym) const;

  /**
   * @brief Sets new interval for symbol
   *
   * @tparam Interval interval template specialization (Integers, Reals)
   * @param sym
   * @param value
   */
  template <class Interval>
  void update_symbol_interval(const symbol2t &sym, const Interval value);

  template <class Interval>
  bool is_mapped(const symbol2t &sym) const;

  template <class Interval>
  expr2tc make_expression_helper(const expr2tc &symbol) const;

  template <class Interval>
  expr2tc make_expression_value(
    const Interval &interval,
    const type2tc &type,
    bool upper) const;

  template <class IntervalMap>
  bool join(IntervalMap &new_map, const IntervalMap &previous_map);
};

#endif // CPROVER_ANALYSES_INTERVAL_DOMAIN_H
