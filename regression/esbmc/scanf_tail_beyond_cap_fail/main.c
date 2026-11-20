#include <assert.h>
#include <stdio.h>

char big[300];

int main()
{
  sscanf("x", "%s", &big[1]);
  assert(big[0] == 0);
  return 0;
}
