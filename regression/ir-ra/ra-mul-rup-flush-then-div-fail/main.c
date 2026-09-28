/* Without interval widening, y's stored interval excludes 0 after flush,
 * making the div constraint UNSAT -- a false-safe verdict. */
extern float __VERIFIER_nondet_float(void);
extern int __ESBMC_rounding_mode;

int main(void)
{
  __ESBMC_rounding_mode = 2; /* FE_UPWARD */
  float x = __VERIFIER_nondet_float();
  __ESBMC_assume(x > 0.0f && x < 1e-22f);
  float y = x * x; /* flushes to zero */
  float z = 1.0f / y; /* division by flushed zero yields +inf */
  __ESBMC_assert(z < 1e30f, "result of div-by-flushed-zero must be finite");
  return 0;
}
