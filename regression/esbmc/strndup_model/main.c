#include <assert.h>
#include <stdlib.h>
#include <string.h>

int main()
{
  char *s = strndup("hello", 2);
  if (s)
  {
    assert(s[0] == 'h' && s[1] == 'e' && s[2] == '\0');
    free(s);
  }

  char *t = strndup("ab", 10);
  if (t)
  {
    assert(t[0] == 'a' && t[1] == 'b' && t[2] == '\0');
    free(t);
  }

  char *e = strndup("xyz", 0);
  if (e)
  {
    assert(e[0] == '\0');
    free(e);
  }

  char raw[3] = {'p', 'q', 'r'};
  char *r = strndup(raw, 3);
  if (r)
  {
    assert(r[2] == 'r' && r[3] == '\0');
    free(r);
  }

  size_t n = nondet_uint();
  __ESBMC_assume(n <= 4);
  char *u = strndup("abc", n);
  if (u)
  {
    assert(u[n < 3 ? n : 3] == '\0');
    if (n > 1)
      assert(u[1] == 'b');
    free(u);
  }
  return 0;
}
