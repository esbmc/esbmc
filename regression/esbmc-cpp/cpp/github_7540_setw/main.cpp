// streamsize is signed ([stream.types]), so setw(-1) leaves a negative width,
// which pads nothing and the insertion resets to 0. With an unsigned
// streamsize the model clamped the argument to 0 instead (#7540).
#include <cassert>
#include <iomanip>
#include <iostream>

int main()
{
  std::cout << std::setw(-1);
  assert(std::cout.width() < 0);
  std::cout << "ab";
  assert(std::cout.width() == 0);
  return 0;
}
