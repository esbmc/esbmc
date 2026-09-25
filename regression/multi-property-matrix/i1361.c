#include <assert.h>
int main()
{
  for (int i = 0; i < 2; i++)
    assert(2 == 3);
  assert(1 == 2);
  return 0;
}
