#ifndef SOLVERS_SMT_FP_IR_IEEE_CONV_H_
#define SOLVERS_SMT_FP_IR_IEEE_CONV_H_

#include <util/message.h>
#include <solvers/smt/smt_ast.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_type.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

class smt_convt;

/** Encapsulates the integer-encoding IEEE 754 path (--int-encoding --ir-ieee).
 *
 *  All state that belongs solely to this encoding is kept here, away from the
 *  main smt_convt class:
 *    - the interval map used for compositional interval lifting
 *    - the set of symbols that have already received range assertions
 *    - the five rounding-mode enclosure helpers
 *    - apply_ieee754_semantics (the single-step enclosure / semantics entry point)
 *    - one encode_ieee_* method per IEEE arithmetic operation
 *
 *  smt_convt holds a pointer to this class (ir_ieee_api) and delegates the
 *  int_encoding branch of each ieee_*_id switch case to the corresponding
 *  encode_ieee_* method.
 */
class ir_ieee_convt
{
public:
  explicit ir_ieee_convt(smt_convt *ctx);

  /** Propagate interval metadata from rhs to lhs after an SSA assignment.
   *  Called from smt_convt::convert_assign so that get_iv() lookups on the
   *  LHS variable find the stored interval for compositional lifting. */
  void propagate_interval(smt_astt lhs, smt_astt rhs);

  /** Assert C integer type range for narrow symbols (width < 32 bits).
   *  Called from smt_convt::convert_terminal for the symbol_id case.
   *  Only asserts when --ir-ieee and int-encoding are both active. */
  void assert_symbol_range(
    const std::string &name,
    smt_astt sym_ast,
    const symbol2t &sym);

  /** Apply IEEE 754 semantics to an exact real arithmetic result.
   *
   *  Under --ir-ieee: asserts a sound rounding enclosure (tight for known
   *  rounding modes, weak for symbolic or unrecognised modes) and returns
   *  real_result unchanged.
   *
   *  Without --ir-ieee: applies overflow / underflow / subnormal clamping for
   *  single and double precision; passes other formats through unchanged. */
  smt_astt apply_ieee754_semantics(
    smt_astt real_result,
    const floatbv_type2t &fbv_type,
    smt_astt operand_zero_check,
    const expr2tc &rounding_mode);

  /** Integer-encoding path for ieee_add.  Returns the SMT AST for the result.
   *  Performs interval-lifted enclosure when --ir-ieee is active and the format
   *  is single or double precision; falls back to apply_ieee754_semantics. */
  smt_astt encode_ieee_add(const expr2tc &expr);

  /** Integer-encoding path for ieee_sub. */
  smt_astt encode_ieee_sub(const expr2tc &expr);

  /** Integer-encoding path for ieee_mul. */
  smt_astt encode_ieee_mul(const expr2tc &expr);

  /** Integer-encoding path for ieee_div. */
  smt_astt encode_ieee_div(const expr2tc &expr);

  /** Integer-encoding path for ieee_fma (fused multiply-add). */
  smt_astt encode_ieee_fma(const expr2tc &expr);

  /** Integer-encoding path for ieee_sqrt. */
  smt_astt encode_ieee_sqrt(const expr2tc &expr);

private:
  smt_convt *ctx;

  /** Interval [lo, hi] in the real-arithmetic SMT encoding. */
  struct ra_interval_t
  {
    smt_astt lo;
    smt_astt hi;
  };

  /** Map from exact-real-result AST pointer to its enclosure interval.
   *  Keyed by pointer identity (SSA variables are hash-consed in smt_cache).
   *  Missing entries fall back to the point interval {t, t}. */
  std::unordered_map<const smt_ast *, ra_interval_t> ir_ra_interval_map;

  /** Set of symbol names that have already received integer range assertions,
   *  preventing duplicate constraints for the same SSA variable. */
  std::unordered_set<std::string> ir_ieee_ranged_syms;

  /** Look up the tracked interval for t; fall back to the point interval. */
  ra_interval_t get_iv(smt_astt t) const;

  /** Dispatch helper: apply the appropriate enclosure based on rounding_mode. */
  std::pair<smt_astt, smt_astt> apply_enclosure(
    smt_astt real_result,
    smt_astt lo_r,
    smt_astt hi_r,
    const floatbv_type2t &fbv_type,
    const expr2tc &rounding_mode);

  std::pair<smt_astt, smt_astt> apply_ieee754_rne_enclosure(
    smt_astt real_result,
    smt_astt lo_r,
    smt_astt hi_r,
    const floatbv_type2t &fbv_type);

  std::pair<smt_astt, smt_astt> apply_ieee754_rna_enclosure(
    smt_astt real_result,
    smt_astt lo_r,
    smt_astt hi_r,
    const floatbv_type2t &fbv_type);

  std::pair<smt_astt, smt_astt> apply_ieee754_rup_enclosure(
    smt_astt real_result,
    smt_astt lo_r,
    smt_astt hi_r,
    const floatbv_type2t &fbv_type);

  std::pair<smt_astt, smt_astt> apply_ieee754_rdn_enclosure(
    smt_astt real_result,
    smt_astt lo_r,
    smt_astt hi_r,
    const floatbv_type2t &fbv_type);

  std::pair<smt_astt, smt_astt> apply_ieee754_rtz_enclosure(
    smt_astt real_result,
    smt_astt lo_r,
    smt_astt hi_r,
    const floatbv_type2t &fbv_type);
};

#endif /* SOLVERS_SMT_FP_IR_IEEE_CONV_H_ */
