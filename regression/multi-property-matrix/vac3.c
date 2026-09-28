#include <assert.h>
int main()
{
  int x = 0;
  while (1)
  {
    assert(x < 3);
    assert(x != 10);
    ++x;
  }
}
