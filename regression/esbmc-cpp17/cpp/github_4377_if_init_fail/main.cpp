#include <cassert>

int main()
{
  // The init-statement initialises x to 7; the assertion below must fail.
  if (int x = 7; x > 0)
  {
    assert(x == 8);
  }
  return 0;
}
