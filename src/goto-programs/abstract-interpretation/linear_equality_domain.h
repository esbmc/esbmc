/// \file
/// Linear Equality Domain
///
/// Implements Karr's algorithm for inferring linear equality invariants.
/// The domain tracks a set of affine equalities of the form
///   x = c0 + c1*y1 + c2*y2 + ...
/// that hold simultaneously at each program point.
///
/// Lattice (top to bottom):
///   top    = {} (empty equation set — no information)
///   ...    = finite sets of consistent equations
///   bottom = contradictory state (unreachable program point)
///
/// Convergence: equations are only removed at merge points, never added,
/// so the fixedpoint terminates in at most O(n) merge steps without widening.

#pragma once

#include <goto-programs/abstract-interpretation/ai.h>
#include <irep2/irep2_utils.h>
#include <util/dstring.h>
#include <util/mp_arith.h>

#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_set>
#include <vector>

/// Linear combination: value = constant + sum_i( coeffs[var_i] * var_i ).
/// Only non-zero coefficients are stored; a missing entry implies zero.
struct AffineExpr
{
  BigInt constant = 0;
  std::map<irep_idt, BigInt> coeffs;

  AffineExpr() = default;
  explicit AffineExpr(BigInt c) : constant(c)
  {
  }
  AffineExpr(irep_idt var, BigInt coeff) : constant(0)
  {
    if (coeff != 0)
      coeffs[var] = coeff;
  }

  AffineExpr operator+(const AffineExpr &rhs) const;
  AffineExpr operator-(const AffineExpr &rhs) const;
  AffineExpr operator*(const BigInt &scalar) const;

  bool is_constant() const
  {
    return coeffs.empty();
  }
  bool operator==(const AffineExpr &rhs) const
  {
    return constant == rhs.constant && coeffs == rhs.coeffs;
  }
  bool operator!=(const AffineExpr &rhs) const
  {
    return !(*this == rhs);
  }

  std::string to_string() const;
};

/// Abstract domain tracking linear equalities between integer variables.
/// The state at each program point is a map var -> AffineExpr meaning
/// "var == AffineExpr" holds on every execution reaching this point.
/// An absent entry means the variable's value is unconstrained (top)..
class linear_equality_domaint : public ai_domain_baset
{
public:
  // Map from variable name to the affine expression it equals.
  // If a key exists: var == equations[var] at this program point.
  // Absent key: variable is unconstrained (top for that variable).
  using equation_map = std::map<irep_idt, AffineExpr>;

  std::shared_ptr<equation_map> equations;
  bool bottom = true; // true when the state is ⊥ (unreachable)

  // Actual irep2 type of each LHS variable, recorded during transform().
  // Used by make_equality_expr to build expressions with the correct bit width.
  std::map<irep_idt, type2tc> symbol_types;

  linear_equality_domaint() : equations(std::make_shared<equation_map>())
  {
  }

  void transform(
    goto_programt::const_targett from,
    goto_programt::const_targett to,
    ai_baset &ai,
    const namespacet &ns) override;

  bool merge(
    const linear_equality_domaint &src,
    goto_programt::const_targett from,
    goto_programt::const_targett to);

  void output(std::ostream &out) const override;

  void make_bottom() override
  {
    equations = std::make_shared<equation_map>();
    bottom = true;
  }
  void make_top() override
  {
    equations = std::make_shared<equation_map>();
    bottom = false;
  }
  void make_entry() override
  {
    make_top();
  }
  bool is_bottom() const override
  {
    return bottom;
  }
  bool is_top() const override
  {
    return !bottom && equations->empty();
  }

  bool ai_simplify(expr2tc &condition, const namespacet &ns) const override;

  /// Convert the full domain state to a conjunctive boolean predicate.
  expr2tc to_predicate() const override;

  /// Convert the domain state to a predicate restricted to [vars].
  /// Only equations whose LHS is in [vars] are included — used during loop
  /// instrumentation to emit only invariants for loop-relevant variables.
  expr2tc
  to_predicate(const std::unordered_set<irep_idt, dstring_hash> &vars) const;

  /// Look up the affine expression known for [var_name], or nullptr if absent.
  const AffineExpr *lookup(const irep_idt &var_name) const;

  size_t size() const
  {
    return equations ? equations->size() : 0;
  }

private:
  void copy_if_needed();
  bool try_as_affine(const expr2tc &expr, AffineExpr &out) const;

  /// Record var = rhs and propagate consistently through other equations.
  /// Three cases:
  ///   A) rhs does not mention var: substitute var_old -> rhs elsewhere.
  ///   B) rhs = var + delta (coefficient 1): invert to var_old = var_new - delta.
  ///   C) rhs mentions var with other coefficient: forget var (non-invertible).
  void assign_affine(const irep_idt &var, const AffineExpr &rhs);

  /// Remove all equations that mention [var] (variable becomes unconstrained).
  void forget(const irep_idt &var);

  /// Substitute every occurrence of [var] in [target] with [expr].
  static AffineExpr substitute(
    const AffineExpr &target,
    const irep_idt &var,
    const AffineExpr &expr);

  /// Build the irep2 expression  lhs_sym == rhs_affine using [lhs_type] for
  /// every sub-expression so all bitvector widths are consistent.
  static expr2tc make_equality_expr(
    const irep_idt &lhs,
    const type2tc &lhs_type,
    const AffineExpr &rhs);
};

using linear_equality_analysist = ait<linear_equality_domaint>;
