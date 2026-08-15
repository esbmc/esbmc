/* Only the decay `&b[0]` names the whole array, so an interior pointer widens
 * to the single element it addresses while the callee writes the tail through
 * p. b[3] keeps its pre-call value and the assertion below holds when it
 * should not. Same wall as github_6961_assigns_ptr_var_frame_knownbug: what is
 * reached through a pointer has no object to widen to. */
#include <assert.h>
#define N 4

void clr(int *p)
{
  __ESBMC_assigns(p);
  __ESBMC_ensures(p[0] == 0);

  for (int i = 0; i < 2; i++)
    p[i] = 0;
}

int main(void)
{
  int b[N];
  b[3] = 7;
  clr(&b[2]);
  assert(b[3] == 7);
  return 0;
}
