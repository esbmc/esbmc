#include <assert.h>
#include <string.h>

unsigned nondet_uint(void);

int main()
{
  char a[100], b[100];
  unsigned n = nondet_uint();
  if (n > sizeof(a))
    return 0;
  memcpy(b, a, n);
  if (n > 70)
    assert(b[70] == a[70]);

  int c[300], d[300];
  unsigned m = nondet_uint();
  if (m > sizeof(c))
    return 0;
  memmove(d, c, m);
  if (m >= 1000)
    assert(d[249] == c[249]);

  int e[10][20], f[10][20];
  unsigned k = nondet_uint();
  if (k > sizeof(e) - 30)
    return 0;
  memcpy((char *)e + 30, f, k);
  if (k > 2)
    assert(((char *)e)[32] == ((char *)f)[2]);
  return 0;
}
