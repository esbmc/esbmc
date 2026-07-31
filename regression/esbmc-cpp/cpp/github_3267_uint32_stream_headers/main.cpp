// libstdc++ 11 (-std=c++17) makes the fixed-width typedefs visible through
// each of these; ESBMC did not, so unmodified sources failed to parse
// (github #3267).
#include <iostream>
#include <ostream>
#include <istream>
#include <streambuf>
#include <iomanip>
#include <iterator>
#include <memory>
#include <cassert>

int main()
{
  uint32_t n = 4000000000u;
  assert(n > 2147483647u);
  assert(sizeof(uint32_t) == 4);
  return 0;
}
