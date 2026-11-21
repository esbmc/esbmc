/* #10888: a struct spanning two elements of an array of structs. */
#include <assert.h>
struct U
{
  unsigned char f1, f2;
};
struct G
{
  struct U f1, f2;
};
unsigned char nondet_uchar(void);
int main(void)
{
  struct U arr[3] = {
    {nondet_uchar(), nondet_uchar()},
    {nondet_uchar(), nondet_uchar()},
    {nondet_uchar(), nondet_uchar()}};
  struct G g = *(struct G *)&arr[1];
  assert(g.f1.f1 == arr[1].f1 && g.f1.f2 == arr[1].f2);
  assert(g.f2.f1 == arr[2].f1 && g.f2.f2 == arr[2].f2);
  return 0;
}
