#include <cassert>

int main()
{
  // The init-statement sets x to 7, so the default branch must be taken
  // and the assertion below must fail.
  switch (int x = 7; x)
  {
  case 0:
    break;
  default:
    assert(0);
  }
  return 0;
}
