/* isqrt ends with cpp::bit_cast<OutType>(r) where r is FracType (u0.32) and
 * OutType is u16.16 -- SAME storage width (32 bits), different scale. A
 * bit_cast must preserve the raw bit pattern, changing only interpretation.
 * The trace showed 0.70711757 (raw 0xB505A837) becoming 0.70710754
 * (raw 0x0000B505), i.e. the pattern was SHIFTED, not reinterpreted. */
unsigned long _Fract nondet_ulr(void);
int main(void)
{
  unsigned long _Fract f = nondet_ulr();   /* u0.32 */
  unsigned int raw_f;
  __ESBMC_bitcast(&raw_f, &f);

  unsigned _Accum a;                        /* u16.16, same 32-bit storage */
  __ESBMC_bitcast(&a, &f);
  unsigned int raw_a;
  __ESBMC_bitcast(&raw_a, &a);

  __ESBMC_assert(raw_a == raw_f, "bitcast u0.32 -> u16.16 preserves the bits");
  return 0;
}
