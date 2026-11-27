/* The negative half of synth_loop_invariant_glibcassert: the same glibc-shaped
 * assert expansion over a bound that is false at a == 0. The synthesised
 * invariant must not mask it: the first iteration from the loop's entry
 * state violates it, so it is a real failure (#11027). */
#include <stdint.h>
extern void
__assert_fail(const char *, const char *, unsigned int, const char *);
#define ASSERT_GLIBC(expr)                                                     \
  ((void)sizeof((expr) ? 1 : 0), __extension__({                               \
     if (expr)                                                                 \
       ;                                                                       \
     else                                                                      \
       __assert_fail(#expr, __FILE__, __LINE__, __func__);                     \
   }))
int main(void)
{
  uint32_t n;
  uint64_t a;
  uint64_t i = 1, sn = 0;
  __ESBMC_assume(n >= 1 && n <= 10);
  __ESBMC_assume(a <= 10);
  while (i <= n)
  {
    ASSERT_GLIBC(sn < (uint64_t)n * a);
    sn = sn + a;
    i++;
  }
  return 0;
}
