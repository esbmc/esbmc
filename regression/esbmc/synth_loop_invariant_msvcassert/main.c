/* MSVC spells assert(e) as `(!!(e)) || (_wassert(...), 0)`, so on Windows a
 * loop body that asserts holds a branch around the ASSERT rather than the
 * single one the glibc and Darwin spellings fold to. The expansion is written
 * out here so the shape is pinned on every host, not only on Windows, where it
 * made the synthesiser decline every loop whose body asserts. */
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
