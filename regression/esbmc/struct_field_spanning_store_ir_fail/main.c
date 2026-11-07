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
struct S s;
int main(void)
{
  unsigned char *p = (unsigned char *)&s;
  *(unsigned long long *)p = 0x0000000200000001ULL;
  assert(s.a.x == 0 && s.b.y == 0);
  return 0;
}
