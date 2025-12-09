#include <cassert>
#include <utility>

int main()
{
  assert((std::make_integer_sequence<int, 20>{}.size()) == 21);
  return 0;
}