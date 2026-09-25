/* The entry jump reaches the back edge only through a failed assertion, so
   it got no havoc and the inductive step "proved" i < 500 (esbmc/esbmc#7900). */
#include <assert.h>
int main()
{
  int i = 0;
  goto L;
  while (i < 1000)
  {
    i = i + 2;
  L:
    __ESBMC_assert(0, "v");
    i = i + 1;
  }
  assert(i < 500);
  return 0;
}
