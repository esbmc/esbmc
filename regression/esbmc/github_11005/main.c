#include <assert.h>
#include <string.h>
unsigned nondet_uint(void);
int main()
{
  char a[100], b[100];
  unsigned n = nondet_uint();
  if (n > 100)
    return 0;
  memcpy(b, a, n);
  if (n > 70)
    assert(b[70] == a[70]);

  char c[100];
  char c5 = c[5], c6 = c[6];
  unsigned m = nondet_uint();
  if (m > 99)
    return 0;
  memmove(c + 1, c, m);
  if (m > 5)
    assert(c[6] == c5);
  else
    assert(c[6] == c6);
  return 0;
}
