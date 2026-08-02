#include <assert.h>
int main()
{
  int x = 1;
  if (x == 0)
    assert(x != 0);
  assert(x == 1);
  return 0;
}
