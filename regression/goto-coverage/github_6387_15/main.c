#include <assert.h>
int main()
{
  int i, s = 0;
  for (i = 0; i < 4; i++)
  {
    if (i > 1)
      s += i;
    assert(s >= 0);
  }
  assert(i == 4);
  return 0;
}
