// #7971: --interval-analysis folds the loop guard away, leaving an ASSERT
// loop head reached by fall-through rather than by a GOTO.
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
