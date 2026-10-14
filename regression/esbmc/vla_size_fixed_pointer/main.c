#include <assert.h>
int main(void)
{
  int n = 3;
  int a[2][n];
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < n; j++)
      a[i][j] = 0;
  a[1][0] = 7;
  int (*p)[n] = a;
  n = 5;
  assert(p[1][0] == 7);
  return 0;
}
