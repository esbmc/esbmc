/* A do-while loop's head is the first body instruction, not the guard, so it
 * is not the `IF !(i <op> B) GOTO exit` shape the recogniser matches against.
 * It declines, silently, and the verdict is unaffected. */
#include <assert.h>

int main(void)
{
  unsigned int n;
  __ESBMC_assume(n >= 1 && n <= 4);

  unsigned int i = 0;
  unsigned int s = 0;

  do
  {
    s = s + 1;
    i = i + 1;
  } while (i < n);

  assert(s == n);
  return 0;
}
