#ifndef SOLVERS_SMT_FP_IR_IEEE_CONV_H_
#define SOLVERS_SMT_FP_IR_IEEE_CONV_H_

#include <solvers/smt/smt_ast.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_type.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

class smt_solver_baset;

/** Encapsulates the integer-encoding IEEE 754 path (--int-encoding --ir-ieee).
 *
 *  All state that belongs solely to this encoding is kept here, away from the
 *  main smt_solver_baset class:
 *    - the interval map used for compositional interval lifting
 *    - the set of symbols that have already received range assertions
 *    - the five rounding-mode enclosure helpers (RNE/RNA/RUP/RDN/RTZ)
 *    - one encode_ieee_* method per supported IEEE arithmetic operation
 *
 *  smt_solver_baset holds a pointer to this class (ir_ieee_api) and delegates
 *  the int_encoding branch of each ieee_*_id switch case to the corresponding
 *  encode_ieee_* method.  The sqrt case and apply_ieee754_semantics remain in
 *  smt_solver_baset and call back into this class via ir_ieee_api. */
class ir_ieee_convt
{
public:
  explicit ir_ieee_convt(smt_solver_baset *ctx);

  /** Interval [lo, hi] in the real-arithmetic SMT encoding. */
  struct ra_interval_t
  {
    smt_astt lo;
    smt_astt hi;
  };

  /** Propagate interval metadata from rhs to lhs after an SSA assignment.
   *  Called from smt_solver_baset::convert_assign. */
  void propagate_interval(smt_astt lhs, smt_astt rhs);

  /** Assert C integer type range for narrow symbols (width < 32 bits).
   *  Called from smt_solver_baset::convert_terminal for symbol_id. */
  void assert_symbol_range(
    const std::string &name,
    smt_astt sym_ast,
    const symbol2t &sym);

  /** Look up the tracked interval for t; fall back to the point interval {t, t}.
   *  Used by both encode_ieee_* methods and the sqrt case in smt_solver.cpp. */
  ra_interval_t get_interval(smt_astt t) const;

  /** Store an interval for t in the map.
   *  Used by both encode_ieee_* methods and the sqrt case in smt_solver.cpp. */
  void store_interval(smt_astt t, smt_astt lo, smt_astt hi);

  /** Integer-encoding path for ieee_add. */
  smt_astt encode_ieee_add(const expr2tc &expr);

  /** Integer-encoding path for ieee_sub. */
  smt_astt encode_ieee_sub(const expr2tc &expr);

  /** Integer-encoding path for ieee_mul. */
  smt_astt encode_ieee_mul(const expr2tc &expr);

  /** Integer-encoding path for ieee_div. */
  smt_astt encode_ieee_div(const expr2tc &expr);

  /** Integer-encoding path for ieee_fma (fused multiply-add). */
  smt_astt encode_ieee_fma(const expr2tc &expr);

  /** Encode ieee_rem2t (IEEE 754 remainder, C's remainder()). Exact, so no
   *  rounding, enclosure or flush; r is pinned as x - n*y through a fresh
   *  integer n nearest x/y, even on ties. */
  smt_astt encode_ieee_rem(const expr2tc &expr);

  /** Record that the SMT AST t may be NaN; nan_pred is a boolean SMT term
   *  that is true iff t holds a NaN value (e.g. not(operand >= 0) for
   *  sqrt with a negative operand). */
  void store_nan_pred(smt_astt t, smt_astt nan_pred);

  /** Return the stored NaN predicate for t, or nullptr if none is known. */
  smt_astt get_nan_pred(smt_astt t) const;

  /** Propagate a NaN predicate from rhs to lhs after an SSA assignment.
   *  Called from smt_solver_baset::convert_assign alongside
   *  propagate_interval. */
  void propagate_nan_pred(smt_astt lhs, smt_astt rhs);

  /** Wrap a comparison result with IEEE NaN semantics.
   *  If either operand has a known NaN predicate, returns
   *    ite(nan_pred, is_neq, cmp)
   *  so that ordered comparisons (is_neq=false) evaluate to false when
   *  either operand is NaN, and != (is_neq=true) evaluates to true.
   *  Returns cmp unchanged when no NaN predicate is known. */
  smt_astt apply_nan_cmp(smt_astt cmp, smt_astt a, smt_astt b, bool is_neq);

  /** Combine two NaN predicates with OR.
   *  Returns nullptr if neither is set; the non-null one if only one is set;
   *  mk_or(a, b) if both are set. */
  smt_astt combine_nan_preds(smt_astt a, smt_astt b) const;

  /** Record a predicate that is true iff the SMT AST t represents
   *  IEEE 754 -0.0. Such predicates originate when a negative
   *  subnormal-range result is flushed to zero and when a literal
   *  negative-zero float constant is converted, and may subsequently
   *  be propagated through assignments or wrapper terms. This does not
   *  implement general signed-zero semantics. */
  void store_neg_zero_pred(smt_astt t, smt_astt neg_zero_pred);

  /** Return the stored negative-zero predicate for t, or nullptr if no
   *  negative-zero metadata is recorded for t. */
  smt_astt get_neg_zero_pred(smt_astt t) const;

  /** Propagate a negative-zero predicate from rhs to lhs after an SSA
   *  assignment. Called from smt_solver_baset::convert_assign alongside
   *  propagate_interval and propagate_nan_pred.
   *
   *  One-way: if rhs carries no predicate, any existing entry already
   *  recorded for lhs is left in place rather than cleared. This is
   *  harmless under SSA, where each assignment gives lhs a fresh AST
   *  with no prior entry of its own, but callers reusing an AST across
   *  more than one assignment should not assume this clears stale
   *  metadata. */
  void propagate_neg_zero_pred(smt_astt lhs, smt_astt rhs);

  /** Re-attach inner's negative-zero predicate (if any) to outer, guarded
   *  by `guard` (the condition under which outer actually evaluates to
   *  inner). Must be used whenever a term that may carry negative-zero
   *  metadata is wrapped in a further SMT term such as an ite -- e.g.
   *  encode_ieee_div's div-by-zero selection -- since the metadata map is
   *  keyed by AST pointer and the wrapper is a different pointer from
   *  inner, so a lookup on the wrapper alone would silently miss it. */
  void propagate_neg_zero_through_ite(
    smt_astt outer,
    smt_astt inner,
    smt_astt guard);

  /** Interval-lifted RNE enclosure helper.
   *  Input: exact real result and pre-computed interval endpoints [lo_r, hi_r].
   *  Returns {ra_lo, ra_hi} for storage in the interval map. */
  std::pair<smt_astt, smt_astt> apply_ieee754_rne_enclosure(
    smt_astt real_result,
    smt_astt lo_r,
    smt_astt hi_r,
    const floatbv_type2t &fbv_type);

  /** Interval-lifted RNA enclosure helper (round-to-nearest-away). */
  std::pair<smt_astt, smt_astt> apply_ieee754_rna_enclosure(
    smt_astt real_result,
    smt_astt lo_r,
    smt_astt hi_r,
    const floatbv_type2t &fbv_type);

  /** Interval-lifted RUP enclosure helper (round-to-plus-infinity). */
  std::pair<smt_astt, smt_astt> apply_ieee754_rup_enclosure(
    smt_astt real_result,
    smt_astt lo_r,
    smt_astt hi_r,
    const floatbv_type2t &fbv_type);

  /** Interval-lifted RDN enclosure helper (round-to-minus-infinity). */
  std::pair<smt_astt, smt_astt> apply_ieee754_rdn_enclosure(
    smt_astt real_result,
    smt_astt lo_r,
    smt_astt hi_r,
    const floatbv_type2t &fbv_type);

  /** Interval-lifted RTZ enclosure helper (round-to-zero / truncation). */
  std::pair<smt_astt, smt_astt> apply_ieee754_rtz_enclosure(
    smt_astt real_result,
    smt_astt lo_r,
    smt_astt hi_r,
    const floatbv_type2t &fbv_type);

private:
  smt_solver_baset *ctx;

  /** Map from exact-real-result AST pointer to its enclosure interval.
   *  Keyed by pointer identity (SSA variables are hash-consed in smt_cache).
   *  Missing entries fall back to the point interval {t, t}. */
  std::unordered_map<const smt_ast *, ra_interval_t> ir_ra_interval_map;

  /** Map from AST pointer to its NaN predicate (a boolean SMT term that is
   *  true iff the value is NaN).  Only populated for sqrt results where
   *  the operand may be negative. */
  std::unordered_map<const smt_ast *, smt_astt> ir_ieee_nan_map;

  /** Map from AST pointer to its negative-zero predicate (a boolean SMT
   *  term that is true iff the value stands for IEEE 754 -0.0). Predicates
   *  originate when a negative subnormal-range result is flushed to zero
   *  (smt_solver_baset::mk_subnormal_flush) and when a literal
   *  negative-zero float constant is converted
   *  (smt_solver_baset::convert_terminal's constant_floatbv_id case), and
   *  may subsequently be propagated through assignments or wrapper terms;
   *  an absent entry means that no negative-zero metadata is recorded for
   *  the value.
   *
   *  Lifetime: keyed on raw AST pointers and never pruned, but
   *  smt_solver_baset::pop_ctx() deletes every AST created since the
   *  matching push_ctx(). A pop can therefore leave both a key and its
   *  stored predicate value dangling; a later lookup that hits such an
   *  entry would hand a freed AST to a consumer. This exposure predates
   *  this map (ir_ieee_nan_map above has the same shape and is subject
   *  to the same limitation) and is not specific to negative-zero
   *  tracking, but is noted here since this map's population grows with
   *  every literal negative-zero constant converted. */
  std::unordered_map<const smt_ast *, smt_astt> ir_ieee_neg_zero_map;

  /** Set of symbol names that have already received integer range assertions,
   *  preventing duplicate constraints for the same SSA variable. */
  std::unordered_set<std::string> ir_ieee_ranged_syms;

  /** Store combine_nan_preds(get_nan_pred(s1), get_nan_pred(s2)) on result.
   *  No-op if neither operand has a known NaN predicate. */
  void store_combined_nan_pred(smt_astt result, smt_astt s1, smt_astt s2);

  /** Returns the max-normal threshold for the given float precision. */
  smt_astt get_max_normal_real(const floatbv_type2t &fbv_type) const;

  /** True iff x > max_normal (positive infinity in the real-arithmetic encoding). */
  smt_astt is_pos_inf_real(smt_astt x, const floatbv_type2t &fbv_type) const;

  /** True iff x < −max_normal (negative infinity in the real-arithmetic encoding). */
  smt_astt is_neg_inf_real(smt_astt x, const floatbv_type2t &fbv_type) const;

  /** True iff |x| > max_normal (either sign of infinity). */
  smt_astt is_inf_real(smt_astt x, const floatbv_type2t &fbv_type) const;

  /** Dispatch the appropriate five-way rounding-mode enclosure. */
  std::pair<smt_astt, smt_astt> apply_enclosure(
    smt_astt real_result,
    smt_astt lo_r,
    smt_astt hi_r,
    const floatbv_type2t &fbv_type,
    const expr2tc &rounding_mode);

  /** Widen [lo, hi] to [min(0,lo), max(0,hi)] so that the zero produced by
   *  mk_subnormal_flush is always within the stored interval. */
  std::pair<smt_astt, smt_astt> widen_for_flush(smt_astt lo, smt_astt hi);
};

#endif /* SOLVERS_SMT_FP_IR_IEEE_CONV_H_ */
