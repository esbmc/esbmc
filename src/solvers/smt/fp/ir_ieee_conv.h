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

  /** Set of symbol names that have already received integer range assertions,
   *  preventing duplicate constraints for the same SSA variable. */
  std::unordered_set<std::string> ir_ieee_ranged_syms;

  /** Store combine_nan_preds(get_nan_pred(s1), get_nan_pred(s2)) on result.
   *  No-op if neither operand has a known NaN predicate. */
  void store_combined_nan_pred(smt_astt result, smt_astt s1, smt_astt s2);

  /** Dispatch the appropriate five-way rounding-mode enclosure. */
  std::pair<smt_astt, smt_astt> apply_enclosure(
    smt_astt real_result,
    smt_astt lo_r,
    smt_astt hi_r,
    const floatbv_type2t &fbv_type,
    const expr2tc &rounding_mode);
};

#endif /* SOLVERS_SMT_FP_IR_IEEE_CONV_H_ */
