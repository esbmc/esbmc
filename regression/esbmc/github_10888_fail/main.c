/* #10888: the struct rebuilt from the object's bytes keeps their order. */
#include <assert.h>
struct T
{
  unsigned char a, b;
};
struct C
{
  unsigned char f1, f2;
};
unsigned char nondet_uchar(void);
int main(void)
{
  struct T s = {nondet_uchar(), nondet_uchar()};
  struct C c = *(struct C *)&s;
  assert(c.f1 == s.b);
  return 0;
}
