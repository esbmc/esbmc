#include <string.h>
/* A weak definition of a libc function the CPROVER additions also link. */
__attribute__((weak)) size_t strlen(const char *s)
{
  (void)s;
  return 99;
}
int main()
{
  const char *t = "abc";
  size_t n = strlen(t);
  __CPROVER_assert(n == 3, "the real strlen wins over the weak stub");
  return 0;
}
