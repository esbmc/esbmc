/* #10888: a struct of array members read through another struct with the
 * same layout and different member names. */
#include <assert.h>
struct G
{
  unsigned char f1[2], f2[2];
};
struct T
{
  unsigned char a[2], b[2];
};
unsigned char nondet_uchar(void);
int main(void)
{
  struct T t = {
    {nondet_uchar(), nondet_uchar()}, {nondet_uchar(), nondet_uchar()}};
  struct G g = *(struct G *)&t;
  assert(g.f1[0] == t.a[0] && g.f1[1] == t.a[1]);
  assert(g.f2[0] == t.b[0] && g.f2[1] == t.b[1]);
  return 0;
}
