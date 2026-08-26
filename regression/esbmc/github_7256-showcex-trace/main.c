#include <assert.h>
unsigned int nondet_uint(void);
int main(void)
{
  unsigned int n = nondet_uint();
  unsigned int i, s = 0;
  for (i = 0; i < n; i++)
    s = s + 1;
  assert(s == n);
  return 0;
}
