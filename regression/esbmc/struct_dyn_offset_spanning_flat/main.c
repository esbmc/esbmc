#include <assert.h>
struct S
{
  unsigned int x, y;
};
struct S s[2];
int main(void)
{
  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i < 2);
  unsigned char *p = (unsigned char *)&s[i];
  *(unsigned long long *)p = 0x0000000200000001ULL;
  assert(s[i].x == 1);
  assert(s[i].y == 2);
  return 0;
}
