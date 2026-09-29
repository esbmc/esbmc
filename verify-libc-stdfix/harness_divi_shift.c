// BUG 2, reduced to the shift itself: fx_bits.h:267 computes `1 << F` where
// F is FRACTION_LEN. For long _Fract / long _Accum (F == 31) the shift
// overflows into the sign bit; for the unsigned variants (F == 32) it shifts
// an int by its full width. Both are undefined behaviour --
// C11 6.5.7p3 (count >= width) and 6.5.7p4 (signed overflow).
//
// The resulting value is used as a DIVISOR one line later, so it is not a
// benign warning: at F == 31 the divisor is negative, which flips the sign of
// every result; at F == 32 it is 1, which drops the scaling entirely.
//
// No fixed-point types are needed to expose this, which is why it is a
// separate C harness -- it is a plain integer defect.
int nondet_int(void);
int main(void)
{
  int f = nondet_int();
  __ESBMC_assume(f == 31 || f == 32); // the FRACTION_LENs that libc reaches
  int scale = 1 << f;                 // fx_bits.h:267
  __ESBMC_assert(scale > 0, "the divisor `1 << F` must be positive");
  return scale;
}
