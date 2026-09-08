/* Regression: GitHub #7670 -- github_7585_weak in MSVC's assert spelling. The
 * invariant leaves the claim open either way, so the verdict must not depend
 * on which <assert.h> the host has: unfolded, the claim is the constant false,
 * which no path can satisfy, and #7585's probe then reads the abstraction as
 * refuting it. */
void _wassert(const char *_Message, const char *_File, unsigned _Line);
#define ASSERT_MSVC(e) \
  (void)((!!(e)) || (_wassert(#e, __FILE__, (unsigned)__LINE__), 0))
int main(void)
{
  int i = 0, s = 0;
  __ESBMC_loop_invariant(i >= 0);
  while (i < 3)
  {
    s++;
    i++;
  }
  ASSERT_MSVC(s == 3);
  return 0;
}
