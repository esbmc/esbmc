/* A proven-safe parallel run: the deciding child carries no assertion
   verdicts, so it reports no table. Pins that the verdict is still correct. */
#include <assert.h>
int main(void)
{
  unsigned n = 0;
  while (n < 10)
    n++;
  assert(n == 10);
  return 0;
}
