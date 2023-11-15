#include <assert.h>

int main()
{
  int i;
  __ESBMC_assume(i > 0 && i < 10);
  if(i > 5)
  {
    assert(0);
    for(int j = 0; j < 100000; j++)
    {
      i = j;
    }
  }
  else
  {
    __ESBMC_assert(1, "");
  }
  return 0;
}
