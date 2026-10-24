// #7971
#include <assert.h>

int main()
{
  int x = 0;
  while (1)
  {
    assert(x >= 0);
    assert(x != 10);
    x = 1;
  }
}
