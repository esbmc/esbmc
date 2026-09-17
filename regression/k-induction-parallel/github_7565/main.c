#include <assert.h>

int main()
{
  int i = 0;
  goto test;
body:
  assert(i < 10);
  i = i + 1;
test:
  if (i < 10)
    goto body;
  return 0;
}
