/* __builtin_ctz* must be modelled, not treated as an undefined function
 * returning nondet. LLVM libc's cpp::countr_zero compiles to __builtin_ctzg,
 * so an unmodelled builtin made every shift count derived from it
 * unconstrained -- ESBMC then reported spurious shift-count violations in
 * anything using it. Same defect class as __builtin_clzg (#6925).
 *
 * Checked against a shift-loop reference over a symbolic input, so this pins
 * correctness rather than merely that the result is bounded. */
unsigned nondet_uint(void);

int main(void)
{
  unsigned v = nondet_uint();
  __ESBMC_assume(v != 0); /* ctz(0) is undefined for the one-argument form */

  int model = __builtin_ctz(v);

  int ref = 0;
  unsigned t = v;
  while ((t & 1u) == 0u)
  {
    t >>= 1;
    ref++;
  }

  __ESBMC_assert(model != ref, "NEGATED");
  __ESBMC_assert(model >= 0 && model < 32, "result is within the operand width");

  /* the generic form's fallback argument makes a zero operand defined */
  __ESBMC_assert(__builtin_ctzg(0u, 32) == 32, "ctzg(0, 32) returns the fallback");
  __ESBMC_assert(__builtin_ctzg(40u, 99) == 3, "ctzg(40, 99) counts, not falls back");
  return 0;
}
