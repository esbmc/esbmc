/* As github_10939_multi_member, but asserting the float write is lost. */
#include <assert.h>
#include <stddef.h>
struct d { unsigned c; };
union u { unsigned char b[4]; unsigned w; float f; };
size_t nondet_size(void);
int main(void)
{
  struct d d = {0};
  size_t off = nondet_size();
  __ESBMC_assume(off == 0);
  union u *p = (union u *)((unsigned char *)&d + off);
  union u n;
  n.f = 1.0f;
  *p = n;
  assert(d.c == 0);
  return 0;
}
