#include <assert.h>
#include <stdio.h>

int main(void)
{
  char b[100];
  char name[4] = {'A', 'B', 'C', 'D'};

  int r = sprintf(b, "%s", name);
  assert(r < 0);
  return 0;
}
