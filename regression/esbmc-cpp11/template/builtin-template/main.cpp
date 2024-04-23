#include <cassert>
#include <utility>

int main()
{
  assert((std::make_integer_sequence<int, 20>{}.size()) == 20);
  assert((std::make_integer_sequence<char, 20>{}.size()) == 20);
  return 0;
}