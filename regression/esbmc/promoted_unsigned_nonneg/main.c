/* C promotes narrow unsigned operands to int before comparison, so the
 * common bounds-check `ref >= count` with u16 operands and count == 0
 * reaches the simplifier as (int)(u16)ref >= 0 — decidably true, since
 * zero-extension never yields a negative. Both spellings appear in the
 * wild: constant on the right (ref >= 0, ref < 0) and on the left
 * (0 <= ref, 0 > ref). Before the fix each guard stayed symbolic and
 * everything behind it stopped folding. */
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
  __ESBMC_assert(n == 3, "constant-right: the refusal folded");

  unsigned short ref2 = nondet_u2();
  int refused2 = 0;
  if (0 <= ref2)
    refused2 = 1;
  int m = 0;
  for (unsigned short i = 0; i < (refused2 ? 3 : 30000); i++)
    m++;
  __ESBMC_assert(m == 3, "constant-left <=: the refusal folded");

  unsigned short ref3 = nondet_u2();
  int refused3 = 0;
  if (!(0 > ref3))
    refused3 = 1;
  int k = 0;
  for (unsigned short i = 0; i < (refused3 ? 3 : 30000); i++)
    k++;
  __ESBMC_assert(k == 3, "constant-left >: the refusal folded");

  unsigned short ref4 = nondet_u2();
  int refused4 = 0;
  if (!(ref4 < count))
    refused4 = 1;
  int j = 0;
  for (unsigned short i = 0; i < (refused4 ? 3 : 30000); i++)
    j++;
  __ESBMC_assert(j == 3, "constant-right <: the refusal folded");
  return 0;
}
