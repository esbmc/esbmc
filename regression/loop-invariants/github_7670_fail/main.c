/* Regression: GitHub #7670 -- github_7585 in MSVC's assert spelling. The
 * invariant pins `sn == n * a` at the exit, which contradicts the claim for
 * every n and a, so folding the branch must not turn a real refutation into
 * an unknown. */
#include <stdint.h>
void _wassert(const char *_Message, const char *_File, unsigned _Line);
#define ASSERT_MSVC(e) \
  (void)((!!(e)) || (_wassert(#e, __FILE__, (unsigned)__LINE__), 0))
int main(void)
{
  uint32_t n;
  uint64_t a;
  uint64_t i = 1, sn = 0;

  __ESBMC_assume(n >= 1);

  __ESBMC_loop_invariant(
    (i <= (uint64_t)n || i == (uint64_t)n + 1) && i >= 1 &&
    sn == (i - 1) * a);

  while (i <= n)
  {
    sn = sn + a;
    i++;
  }

  ASSERT_MSVC(sn == (uint64_t)n * a + 1);
  return 0;
}
