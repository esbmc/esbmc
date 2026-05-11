#include <cassert>

int main()
{
  bool taken = false;
  switch (int x = 7; x)
  {
  case 7:
    assert(x == 7);
    taken = true;
    break;
  default:
    assert(0);
  }
  assert(taken);
  return 0;
}
