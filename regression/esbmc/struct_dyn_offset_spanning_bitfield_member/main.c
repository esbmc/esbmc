#include <assert.h>
struct S
{
  unsigned int a : 12, b : 20;
  unsigned int c;
  int *q;
};
struct S s[2];
int g;
int main(void)
{
  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i < 2);
  s[i].q = &g;
  unsigned int *pc = &s[i].c;
  *pc = 7;
  *s[i].q = 4;
  assert(g == 4 && s[i].c == 7);
  return 0;
}
