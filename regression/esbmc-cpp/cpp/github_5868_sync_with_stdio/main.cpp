#include <iostream>
#include <cassert>
int main()
{
  // [ios.base.sync]/2: the initial state is synchronised.
  assert(std::ios_base::sync_with_stdio(false) == true);
  // ... and each call reports the state the previous one installed.
  assert(std::ios_base::sync_with_stdio(true) == false);
  assert(std::ios_base::sync_with_stdio(true) == true);
  // Reachable through a stream object too, and through std::ios.
  assert(std::cout.sync_with_stdio(false) == true);
  assert(std::ios::sync_with_stdio() == false);
  return 0;
}
