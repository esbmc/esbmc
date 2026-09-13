/* glibc spells assert(e) with a leading `(void) sizeof ((e) ? 1 : 0)`, which
 * lands inside the branch region as an OTHER; Darwin's spelling has no such
 * subexpression, so synth_loop_invariant_condassert exercises the region only
 * on a glibc host. Write the expansion out so the region's inert set is pinned
 * on every platform. */
#include <stdint.h>

extern void
__assert_fail(const char *, const char *, unsigned int, const char *);

int main(void)
{
  uint32_t n;
  uint64_t a;
  uint64_t i = 1, sn = 0;
  __ESBMC_assume(n >= 1 && n <= 10);
  __ESBMC_assume(a <= 10);
  while (i <= n)
  {
    if (i > 1)
      ((void)sizeof((sn >= a) ? 1 : 0), __extension__({
         if (sn >= a)
           ;
         else
           __assert_fail("sn >= a", "main.c", 22, __func__);
       }));
    sn = sn + a;
    i++;
  }
  return 0;
}
