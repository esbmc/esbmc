// The operand is unconstrained, not unusable: properties that hold for every
// input still verify, and the stream stays in a usable state.
#include <iostream>
#include <cassert>

int main()
{
  int i = 0;
  std::cin >> i;
  if (i > 0)
    assert(i >= 1);

  unsigned u = 0;
  std::cin >> u;
  assert(u >= 0);
  return 0;
}
