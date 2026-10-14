/* The failing half of msvc_assert_fold: the fold must keep the claim and its
 * message, not swallow it with the branch it collapses. */
void _wassert(const char *_Message, const char *_File, unsigned _Line);
#define ASSERT_MSVC(e) \
  (void)((!!(e)) || (_wassert(#e, __FILE__, (unsigned)__LINE__), 0))
int main(void)
{
  unsigned n;
  __ESBMC_assume(n < 4);
  ASSERT_MSVC(n <= 2);
  return 0;
}
