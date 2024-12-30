#include <iostream>
#include <cassert>

int main()
{
  assert(
    std::cerr.__streambuf_state ==
    std::ios_base::goodbit); // This is done by the constructor for iod_base.
  return 0;
}
