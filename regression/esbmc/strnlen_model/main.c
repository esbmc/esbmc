#include <assert.h>
#include <string.h>

int main()
{
  char buf[4] = {'a', 'b', 'c', 'd'};
  assert(strnlen("hello", 3) == 3);
  assert(strnlen("hi", 10) == 2);
  assert(strnlen("hello", 0) == 0);
  assert(strnlen(buf, 4) == 4);

  char s[6] = "abcde";
  unsigned n = nondet_uint();
  __ESBMC_assume(n <= 5);
  s[n] = '\0';
  assert(strnlen(s, 6) == n);
  return 0;
}
