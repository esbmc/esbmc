extern double __VERIFIER_nondet_double(void);

int main(void)
{
  /* IEEE 754: a literal -0.0 constant has its sign bit set, so
   * signbit(-0.0) must be nonzero. q = x / z gives z a use beyond the
   * signbit check itself; whether that is enough to keep z's conversion
   * off ESBMC's own constant-folding path is build/platform dependent
   * and not something this test asserts either way -- it only asserts
   * the IEEE 754 semantic property.
   *
   * Use __builtin_signbit rather than <math.h>'s signbit: on some
   * platforms (e.g. macOS) the latter expands to a libm inline that does
   * its own bit-twiddling and never reaches convert_signbit, so a test
   * written with signbit would silently fail to exercise this path. */
  double x = __VERIFIER_nondet_double();
  __ESBMC_assume(x > 0.0);
  double z = -0.0;
  double q = x / z;
  (void)q;
  __ESBMC_assert(
    __builtin_signbit(z) != 0,
    "a literal -0.0 constant must report a negative sign bit");
  return 0;
}
