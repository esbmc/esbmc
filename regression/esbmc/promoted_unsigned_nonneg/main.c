/* C promotes narrow unsigned operands to int before comparison, so the
 * common bounds-check `ref >= count` with u16 operands and count == 0
 * reaches the simplifier as (int)(u16)ref >= 0 — decidably true, since
 * zero-extension never yields a negative. Before the fix the guard
 * stayed symbolic and everything behind it stopped folding. */
unsigned short nondet_u2(void);

static unsigned short count; /* zero */

int main(void)
{
  unsigned short ref = nondet_u2();
  int refused = 0;
  if (ref >= count)
    refused = 1;
  int n = 0;
  for (unsigned short i = 0; i < (refused ? 3 : 30000); i++)
    n++;
  __ESBMC_assert(n == 3, "the refusal folded and bounded the loop");
  return 0;
}
