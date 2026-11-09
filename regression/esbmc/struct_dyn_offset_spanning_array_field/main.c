#include <assert.h>
struct S
{
  unsigned char c[6];
  unsigned short h;
};
struct S s[2];
int main(void)
{
  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i < 2);
  unsigned char *p = (unsigned char *)&s[i] + 4;
  *(unsigned int *)p = 0x04030201u;
  assert(s[i].c[4] == 1 && s[i].c[5] == 2 && s[i].h == 0x0403);
  return 0;
}
