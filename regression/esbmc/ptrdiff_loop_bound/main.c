#include <assert.h>

int main()
{
  int a[5];
  int *p = &a[0], *q = &a[4];

  assert(q - p == 4);
  assert(p - q == -4);

  unsigned n = (unsigned)(q - p);
  unsigned s = 0;
  for (unsigned i = 0; i < n; i++)
    s++;
  assert(s == 4);
  return 0;
}
