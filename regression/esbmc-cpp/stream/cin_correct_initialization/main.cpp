#include <iostream>
#include <cassert>

int main()
{
  assert(
    std::cin.__streambuf_state ==
    std::ios_base::goodbit); // This is done by the constructor for ios_base.
  return 0;
}
