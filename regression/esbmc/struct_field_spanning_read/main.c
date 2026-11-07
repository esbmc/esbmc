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
struct T
{
  unsigned char pad, a, b, tail;
};
struct S s = {{1}, {2}};
struct T t = {9, 5, 7, 8};
int main(void)
{
  unsigned char *p = (unsigned char *)&s;
  assert(*(unsigned long long *)p == 0x0000000200000001ULL);
  unsigned char *q = (unsigned char *)&t;
  assert(*(unsigned short *)(q + 1) == 0x0705);
  return 0;
}
