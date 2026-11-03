#include <assert.h>
int main(void)
{
  int x = 0;
  if (x)
    assert(0);
  assert(x == 0);
  return 0;
}
