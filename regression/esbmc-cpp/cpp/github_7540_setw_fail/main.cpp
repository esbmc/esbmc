// setw(-1) sets width() to -1, not 0; the clamp the unsigned streamsize needed
// made this assertion provable (#7540).
#include <cassert>
#include <iomanip>
#include <iostream>

int main()
{
  std::cout << std::setw(-1);
  assert(std::cout.width() == 0);
  return 0;
}
