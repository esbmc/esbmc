/* The positive half of github_7009_self_assigned_arg_fail: what the contract
 * does imply about the same call still holds, so the result is written back
 * rather than merely left unconstrained. */
#include <assert.h>

int f(int x)
{
  __ESBMC_requires(x > 0);
  __ESBMC_ensures(__ESBMC_return_value == x + 1);
  return x + 1;
}

int main(void)
{
  int r = 5;
  r = f(r);
  assert(r == 6);
  return 0;
}
