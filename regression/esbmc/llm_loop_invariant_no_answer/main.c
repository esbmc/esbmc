/* The model command fails every time: each round is skipped with a warning
 * and the run goes on without invariants, unwinding the bounded loop. */
#include <assert.h>

int main(void)
{
  unsigned i = 0, x = 0;
  while (i < 4)
  {
    x += 2;
    i++;
  }
  assert(x == 8);
  return 0;
}
