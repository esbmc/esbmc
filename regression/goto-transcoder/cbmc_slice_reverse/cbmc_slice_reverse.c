#include <stddef.h>
static void reverse(int *a, size_t n)
{
  for (size_t i = 0; i < n / 2; i++)
  {
    int t = a[i];
    a[i] = a[n - 1 - i];
    a[n - 1 - i] = t;
  }
}
int main()
{
  int a[5] = {1, 2, 3, 4, 5};
  reverse(a, 5);
  __CPROVER_assert(a[0] == 5 && a[4] == 1, "ends swapped");
  __CPROVER_assert(a[2] == 3, "middle fixed");
  return 0;
}
