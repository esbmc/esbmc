#include <assert.h>

/* Byte k of a struct of 16-bit members, at a nondet k. */
struct P
{
  unsigned short x, y;
};

int main()
{
  struct P p;
  p.x = nondet_ushort();
  p.y = nondet_ushort();
  unsigned k = nondet_uint();
  __ESBMC_assume(k < 4);
  unsigned char *c = (unsigned char *)&p;
  unsigned short v = k < 2 ? p.x : p.y;
  assert(c[k] == (unsigned char)(k % 2 == 0 ? v >> 8 : v));
  return 0;
}
