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
  int a[6] = {1, 2, 3, 4, 5, 6};
  size_t n;
  __CPROVER_assume(n >= 1 && n <= 6);
  int first = a[0], last = a[n - 1];
  reverse(a, n);
  __CPROVER_assert(a[0] == last && a[n - 1] == first, "symbolic-length reverse swaps the ends");
  return 0;
}
