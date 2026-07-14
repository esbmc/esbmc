#include <stddef.h>

/* Both directions of <= holding simultaneously for pointers to two distinct
 * objects would violate antisymmetry; asserting that they do must fail. */
int main()
{
  char a, b;
  char *p1 = &a, *p2 = &b;
  int le = (p1 <= p2);
  int ge = (p2 <= p1);
  __ESBMC_assert(le && ge, "both directions cannot hold for distinct objects");
  return 0;
}
