/* The user's invariant calls count_to(), whose own loop is exactly the shape
 * the recogniser accepts. Synthesising there would cut that loop, and the
 * marker would then read a havoc-abstracted return value -- including in the
 * base case, the one obligation evaluated at the concrete pre-loop state. A
 * user invariant is authoritative, so synthesis stands down in every function
 * it calls. */
#include <assert.h>

static unsigned int count_to(unsigned int n)
{
  unsigned int i = 0;
  unsigned int s = 0;
  while (i < n)
  {
    s = s + 1;
    i = i + 1;
  }
  return s;
}

int main(void)
{
  unsigned int i = 0;

  __ESBMC_loop_invariant(i <= 4);
  __ESBMC_loop_invariant(count_to(4) == 4);
  while (i < 4)
    i++;

  assert(i == 4);
  return 0;
}
