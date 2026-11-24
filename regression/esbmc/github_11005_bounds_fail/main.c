#include <string.h>
unsigned nondet_uint(void);
int main()
{
  char a[100], b[100];
  unsigned n = nondet_uint();
  if (n > 101)
    return 0;
  memcpy(b, a, n);
  return 0;
}
