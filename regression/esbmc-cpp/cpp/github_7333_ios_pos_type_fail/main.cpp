#include <iostream>
#include <cassert>

int main()
{
  std::ios::pos_type bad = std::ios::pos_type(-1);
  std::ios::off_type off = std::ios::off_type(3);
  std::ios::pos_type p = std::ios::pos_type(0) + off;
  assert(p == bad);
  return 0;
}
