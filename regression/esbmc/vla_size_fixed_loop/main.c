#include <assert.h>
int main(void)
{
  int n = 1;
  for (int k = 1; k <= 3; k++)
  {
    int a[n];
    n = n + 1;
    assert(sizeof(a) == k * sizeof(int));
  }
  return 0;
}
