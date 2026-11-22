#include <assert.h>
struct S
{
  unsigned int a, b, c;
};
struct S s[4];
int main(void)
{
  int i = nondet_int();
  __ESBMC_assume(i == 1 || i == 3);
  unsigned char *p = (unsigned char *)&s[i] + 4;
  *(unsigned long long *)p = 0x0000000200000001ULL;
  assert(s[i].b == 1);
  assert(s[i].c == 0);
  return 0;
}
