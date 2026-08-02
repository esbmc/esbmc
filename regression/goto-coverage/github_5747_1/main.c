#include <assert.h>

int main()
{
  int a = 2;
  __ESBMC_assert(!(a % 2), "a stays even");
  assert(a == 2);
  return 0;
}
