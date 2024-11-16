#include <assert.h>
int main()
{
  unsigned n = nondet_uint();
  char a[n];
  unsigned k = sizeof(a);
  assert(n == k);
}
