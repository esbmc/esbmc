#include <assert.h>
#include <stdio.h>

int main()
{
  char buf[8] = {0};
  char fmt[3] = {'%', 's', 0};
  int r = sscanf("x", fmt, buf);
  if (r == 1)
    assert(buf[5] == 0);
  return 0;
}
