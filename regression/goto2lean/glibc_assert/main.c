/* glibc's assert expansion, written out so every host sees its leading
 * `(void) sizeof (...)` expression statement. */
extern void
__assert_fail(const char *, const char *, unsigned int, const char *);
#define ASSERT_GLIBC(expr)                                                     \
  ((void)sizeof((expr) ? 1 : 0), __extension__({                               \
     if (expr)                                                                 \
       ;                                                                       \
     else                                                                      \
       __assert_fail(#expr, __FILE__, __LINE__, __func__);                     \
   }))
unsigned nondet_uint(void);
unsigned mid(unsigned lo, unsigned hi)
{
  return (lo + hi) / 2;
}
int main(void)
{
  unsigned lo = nondet_uint(), hi = nondet_uint();
  __ESBMC_assume(lo <= hi);
  unsigned m = mid(lo, hi);
  ASSERT_GLIBC(lo <= m && m <= hi);
  return 0;
}
