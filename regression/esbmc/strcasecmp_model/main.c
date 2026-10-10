#include <assert.h>
#include <strings.h>

int main(void)
{
  assert(strcasecmp("Hello", "hELLO") == 0);
  assert(strcasecmp("abc", "ABD") < 0);
  assert(strcasecmp("abc", "AB") > 0);
  assert(strcasecmp("[", "a") < 0);
  assert(strncasecmp("abcX", "ABCy", 3) == 0);
  assert(strncasecmp("abcX", "ABCy", 4) < 0);
  assert(strncasecmp("x", "y", 0) == 0);

  char c;
  __ESBMC_assume(c >= 'A' && c <= 'Z');
  char s[2] = {c, 0};
  char t[2] = {c + ('a' - 'A'), 0};
  assert(strcasecmp(s, t) == 0);
  return 0;
}
