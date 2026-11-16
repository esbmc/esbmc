#include <assert.h>
#include <string.h>
struct S
{
  char name[8];
  int x;
} s;
int main(void)
{
  s.x = 5;
  unsigned n = nondet_uint();
  __ESBMC_assume(n <= sizeof s);
  memset(&s, 0, n);
  assert(s.x == 5);
  return 0;
}
