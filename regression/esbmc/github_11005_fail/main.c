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
  if (n >= 70)
    assert(b[70] == a[70]);
  return 0;
}
