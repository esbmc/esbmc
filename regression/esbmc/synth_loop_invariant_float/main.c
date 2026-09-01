/* A floating-point counter. The closed form arithmetises over the counter's
 * type, and s0 + (i - i0) * e over a float is neither exact nor what the
 * establishment argument assumes, so the loop must not be summarised and the
 * verdict must be unaffected. Pinned as an observable contract: the recogniser
 * currently rejects this twice over (the integer gate, and the counter's
 * addend not being the literal 1), so the test survives either guard alone. */
#include <assert.h>
int main(void)
{
  double x = 0.0;
  int k = 0;
  while (x < 3.0)
  {
    x = x + 1.0;
    k = k + 1;
  }
  assert(k == 3);
  return 0;
}
