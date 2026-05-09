#include <cassert>

int main()
{
  int outer = 1;

  if (int x = 7; x > 0)
  {
    assert(x == 7);
  }

  // The init-statement variable must not leak out of the if.
  assert(outer == 1);
  return 0;
}
