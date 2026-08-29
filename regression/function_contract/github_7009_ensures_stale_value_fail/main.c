#include <assert.h>

int f(void)
{
  __ESBMC_ensures(__ESBMC_return_value >= 0);
  return 7;
}

int main(void)
{
  int r = 200;
  r = f();
  /* The contract permits any non-negative result, so the caller cannot know
     the value is still 200. Reported SUCCESSFUL while the return value was
     never havocked (#7009). */
  assert(r == 200);
  return 0;
}
