// setprecision(-1) leaves a negative precision; the clamp the unsigned
// streamsize needed made this assertion provable (#7540).
#include <cassert>
#include <iomanip>
#include <iostream>

int main()
{
  std::cout << std::setprecision(-1);
  assert(std::cout.precision() >= 0);
  return 0;
}
