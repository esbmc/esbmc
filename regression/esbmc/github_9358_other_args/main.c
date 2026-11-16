// #9358: --printf-check claims no NUL for a non-%s conversion or for a %s
// argument that may point into either of two arrays.
#include <stdio.h>

_Bool nondet_bool(void);

int main(void)
{
  char a[3] = {'a', 0, 0};
  char b[2] = {'b', 0};
  char *s = nondet_bool() ? a : b;
  long n = 7;
  printf("%d %ld %s\n", 1, n, s);
  return 0;
}
