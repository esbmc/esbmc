#include <assert.h>

int main(void)
{
  int a[2][3][4];
  int n = 0;

  for (int *p = &a[0][0][0]; p != &a[1][2][3]; ++p)
    ++n;

  assert(n == 22);
  return 0;
}
