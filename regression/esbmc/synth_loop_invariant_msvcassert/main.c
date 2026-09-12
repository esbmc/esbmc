/* MSVC spells assert(e) as `(!!(e)) || (_wassert(...), 0)`, which once left a
 * branch around the ASSERT and so made the synthesiser decline every loop whose
 * body asserts on Windows. Since #7671 lowers a discarded `||` as a statement
 * the body folds to a single top-level ASSERT, as the glibc spelling does
 * (glibc leaves an OTHER beside it, which synth_loop_invariant_glibcassert
 * pins). What this pins now is that the fold keeps the loop recognisable on
 * every host; synth_loop_invariant_condassert covers the branch shape that
 * survives lowering. */
#include <stddef.h>
#include <stdint.h>
void _wassert(const wchar_t *_Message, const wchar_t *_File, unsigned _Line);
#define WIDEN_(x) L##x
#define WIDEN(x) WIDEN_(x)
#define ASSERT_MSVC(e) \
  (void)((!!(e)) || (_wassert(WIDEN(#e), WIDEN(__FILE__), (unsigned)__LINE__), 0))
int main(void)
{
  uint32_t n;
  uint64_t a;
  uint64_t i = 1, sn = 0;
  __ESBMC_assume(n >= 1 && n <= 10);
  __ESBMC_assume(a <= 10);
  while (i <= n)
  {
    ASSERT_MSVC(sn <= (uint64_t)n * a);
    sn = sn + a;
    i++;
  }
  return 0;
}
