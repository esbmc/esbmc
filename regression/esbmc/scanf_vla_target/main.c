#include <assert.h>
#include <stdio.h>

int main()
{
  unsigned n;
  __ESBMC_assume(n == 4);
  char buf[n];
  buf[0] = 'a';
  int r = sscanf("xy", "%s", &buf[1]);
  if (r == 1)
    assert(buf[0] == 'a');
  return 0;
}
