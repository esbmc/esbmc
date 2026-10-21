/* A bitcast between two fixed-point formats of the same storage width must
 * reinterpret the bits, not rescale. ESBMC used to fall through to a
 * value-preserving typecast here, so a u0.32 pattern read back as u16.16 came
 * out shifted by the 16-bit difference in fraction length.
 *
 * LLVM libc's fixed_point::isqrt ends in exactly this cast, so the rescale
 * silently corrupted every result computed through it. */
unsigned long _Fract nondet_ulr(void);

int main(void)
{
  unsigned long _Fract f = nondet_ulr(); /* u0.32 */
  unsigned int raw_f;
  __ESBMC_bitcast(&raw_f, &f);

  unsigned _Accum a; /* u16.16 -- same 32-bit storage, different scale */
  __ESBMC_bitcast(&a, &f);
  unsigned int raw_a;
  __ESBMC_bitcast(&raw_a, &a);

  __ESBMC_assert(raw_a == raw_f, "bitcast u0.32 -> u16.16 preserves the bits");
  return 0;
}
