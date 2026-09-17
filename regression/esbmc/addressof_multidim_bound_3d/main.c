// Three dimensions pin the stride product: the offset &a[1][2][3] linearises
// to 1*12 + 2*4 + 3, so a per-level stride that stopped at one dimension would
// place the exit somewhere else and change the trip count.
#include <assert.h>

int main(void)
{
  int a[2][3][4];
  int n = 0;

  for (int *p = &a[0][0][0]; p != &a[1][2][3]; ++p)
    ++n;

  assert(n == 23);
  return 0;
}
