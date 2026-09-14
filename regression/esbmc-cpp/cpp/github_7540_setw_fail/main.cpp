// setw(-1) leaves a negative width. An unsigned streamsize, or the clamp it
// needed, made this assertion provable (#7540).
#include <cassert>
#include <iomanip>
#include <iostream>

int main()
{
  std::cout << std::setw(-1);
  assert(std::cout.width() >= 0);
  return 0;
}
