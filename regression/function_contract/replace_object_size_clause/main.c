#include <stddef.h>
static size_t qv;
size_t mylen(const char *s)
{
  __ESBMC_requires(__ESBMC_exists(&qv, qv < __ESBMC_get_object_size(s) - __ESBMC_POINTER_OFFSET(s) && s[qv] == 0));
  __ESBMC_ensures(__ESBMC_return_value < __ESBMC_get_object_size(s) - __ESBMC_POINTER_OFFSET(s) && s[__ESBMC_return_value] == 0);
  __ESBMC_ensures(__ESBMC_forall(&qv, !(qv < __ESBMC_return_value) || s[qv] != 0));
  __ESBMC_assigns();
  size_t n = 0;
  while (s[n]) n++;
  return n;
}
int main()
{
  char b[4] = {'a', 'b', 0, 'c'};
  size_t n = mylen(b);
  __ESBMC_assert(n == 2, "len");
  char c[2] = {'a', 'b'};
  mylen(c);
}
