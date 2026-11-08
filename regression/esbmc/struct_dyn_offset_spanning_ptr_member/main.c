#include <assert.h>
struct S
{
  int *q;
  unsigned int a, b;
};
struct S s[2];
int g;
int main(void)
{
  s[0].q = &g;
  s[1].q = &g;
  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i < 2);
  unsigned char *p = (unsigned char *)&s[i].a;
  *(unsigned long long *)p = 0x0000000200000001ULL;
  *s[i].q = 7;
  assert(g == 7);
  assert(s[i].b == 2);
  return 0;
}
