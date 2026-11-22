#include <assert.h>
struct A
{
  unsigned int x;
};
struct B
{
  unsigned int y;
};
struct S
{
  struct A a;
  struct B b;
};
struct S s[2] = {{{1}, {2}}, {{1}, {2}}};
int main(void)
{
  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i < 2);
  unsigned char *p = (unsigned char *)&s[i];
  assert(*(unsigned long long *)p == 0x0000000200000001ULL);
  return 0;
}
