/* #10046: a store must not reach t. */
#include <assert.h>
struct In
{
  unsigned char x;
};
struct Out
{
  unsigned char p;
  struct In in;
  unsigned char q;
  unsigned char t;
};
struct Out s;
int main(void)
{
  unsigned char *p = (unsigned char *)&s;
  *(unsigned short *)(p + 1) = 0x0705;
  assert(s.in.x == 5);
  assert(s.q == 7);
  assert(s.t == 7);
  return 0;
}
