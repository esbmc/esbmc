#include <assert.h>
union A
{
  unsigned int x;
  unsigned char c[4];
};
union B
{
  unsigned int y;
  unsigned short h[2];
};
struct S
{
  union A a;
  union B b;
};
struct S s[2];
int main(void)
{
  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i < 2);
  unsigned char *p = (unsigned char *)&s[i];
  *(unsigned long long *)p = 0x0000000200000001ULL;
  assert(s[i].a.x == 1);
  assert(s[i].b.y == 2);
  return 0;
}
