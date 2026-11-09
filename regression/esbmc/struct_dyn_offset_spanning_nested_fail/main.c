#include <assert.h>
struct T
{
  unsigned short a, b, c;
};
struct S
{
  struct T t;
  unsigned short d;
};
struct S s[2];
int main(void)
{
  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i < 2);
  unsigned char *p = (unsigned char *)&s[i] + 4;
  *(unsigned int *)p = 0x04030201u;
  assert(s[i].t.c == 0x0201 && s[i].d == 0);
  return 0;
}
