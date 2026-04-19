/* Regression test: RTZ (ROUND_TO_ZERO) interval lifting for ieee_mul --
 * both operands tracked, double precision.
 *
 * PURPOSE
 * -------
 * Verifies that when both operands of a second RTZ ieee_mul were themselves
 * results of a prior tracked RTZ ieee_mul, ir_ra_interval_map lookup fires
 * for both operands and the interval-lifted RTZ multiplication path is taken.
 *
 * PROOF SHAPE (B_dir, RTZ, double precision)
 * ------------------------------------------
 * First mul:  z = x * y   (both fresh -> point-interval fallback)
 *   ra_lo_tz::0, ra_hi_tz::0 pinned via RTZ three-way ITE on sign of real_z
 *   stored: ir_ra_interval_map[real_z] = {ra_lo_tz::0, ra_hi_tz::0}
 *
 * Second mul:  w = z * z  (both operands are z -> both tracked)
 *   iv(z) = {ra_lo_tz::0, ra_hi_tz::0}
 *   p1 = ra_lo_tz::0 * ra_lo_tz::0
 *   p2 = ra_lo_tz::0 * ra_hi_tz::0
 *   p3 = ra_hi_tz::0 * ra_lo_tz::0
 *   p4 = ra_hi_tz::0 * ra_hi_tz::0
 *   lo_r = min(p1,p2,p3,p4) via nested ITE
 *   hi_r = max(p1,p2,p3,p4) via nested ITE
 *   ra_lo_tz::1, ra_hi_tz::1 pinned via RTZ three-way ITE on sign of [lo_r,hi_r]
 *   Since z spans an interval crossing zero (ra_lo_tz::0 <= z <= ra_hi_tz::0),
 *   z*z can produce a hull that also crosses zero symbolically, triggering the
 *   B_dir_max conservative crossing-zero fallback.
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo_tz::0   -- first mul's lower bound declared
 *   ra_lo_tz::1   -- second mul's lifted lower bound declared
 *   (* |smt_conv::ra_lo_tz::0|  -- endpoint product in hull computation
 *   (ite           -- nested ITE for hull and RTZ three-way sign check
 *   22204460492503131  -- Z3 numerator for eps_rel_dir = 2^-52 (double)
 *   ^VERIFICATION FAILED$
 */
#include <assert.h>

extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 4; /* ROUND_TO_ZERO */
  double x = __VERIFIER_nondet_double();
  double y = __VERIFIER_nondet_double();
  double z = x * y; /* first RTZ mul: both fresh -> point fallback; stored */
  double w = z * z; /* second RTZ mul: both operands tracked -> full lift */

  /* Always false in real/integer encoding: w == z * z exactly. */
  assert(w != z * z);
  return 0;
}
