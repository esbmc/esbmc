#include <iostream>
#include <cassert>

int main()
{
  std::ios::pos_type bad = std::ios::pos_type(-1);
  assert(bad == std::ios::pos_type(-1));

  std::ios::off_type off = std::ios::off_type(4) - std::ios::off_type(1);
  assert(off == 3);

  std::ios::pos_type p = std::ios::pos_type(0) + off;
  assert(p == std::ios::pos_type(3));
  assert(p != bad);

  std::ios::pos_type here = std::cin.tellg();
  assert(here == here);

  std::ios::char_type c = 'a';
  std::ios::int_type i = std::ios::traits_type::to_int_type(c);
  assert(i == 97);
  assert(std::ios::traits_type::to_char_type(i) == c);

  return 0;
}
